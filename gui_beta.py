import asyncio
import sys
import threading

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QMessageBox, 
    QVBoxLayout, QHBoxLayout, QGridLayout, QPlainTextEdit, QFrame, QRadioButton, QButtonGroup, QTextEdit, QSplitter,QFileDialog, QDialog
)
from PyQt6.QtCore import Qt, pyqtSignal, QEvent, QSize, QRect, QUrl
from PyQt6.QtGui import QKeySequence, QShortcut, QPainter, QPen
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtMultimedia import QMediaPlayer

from RAG.rag import RAG_pipeline
from fpdf import FPDF
import shutil
import os

from stt_functions.stt_recognition import listen_and_recognize_multi, stop_listening_event, stt_for_query

from stt_functions.speech_realtime_unlimit_sep1 import  stop_recognition_sep, continuous_recognition_sep

# from stt_functions.speech_realtime_unlimit import recog_stream, stop_recognition, continuous_recognition
from stt_functions.speech_realtime_emotion import stop_recognition, continuous_recognition
from stt_functions.content_correct import content_correct
from answer_request.llm_direct_response import llm_direct_response
import shutil  
import file_conversion.Imagett as Imagett
import file_conversion.office_2_pdf as office_2_pdf
from file_conversion.video_process_async import process_video_combined_async

import pyttsx3  # Text-to-Speech library
from utils.utils_beta import (append_system_message, append_recognition_speaker,append_recognition_user, append_recognition, clear_source_directory, save_transcription, init_ui, get_videorag_answer,
                                OverlayWidget, VideoPreviewDialog, upload_mp4_with_roi, append_Video, append_conversation, clean_server_workdir)

from file_conversion.pdf_split import extract_text_and_images
from file_conversion.video_process_realtime import VideoProcessingThread
from file_conversion.audio_in_video_real_unlimit import RealTimeStreamingTranscriptionThread
from stt_functions.openai_tts import speak_text_thread
from answer_request.video_summary import VideoSummaryThread
from VideoRAG.Video_pipeline import VideoUploadThread
import requests


class ConvoAid(QWidget):
    """Conversational Aid System GUI using PyQt6."""
    # add signal used to append recognition message
    # use the signal to trigger the submit question function(from sub-thread to main thread)

    recognized_text_Signal_stt = pyqtSignal(str)  # Signal to update the recognized text in the UI
    recognized_text_Signal_dialog = pyqtSignal(str)
    submitQuestionSignal = pyqtSignal()   
    
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Conversational Aid System")
        self.setGeometry(100, 100, 1000, 750)

        self.current_chunk_number = 0  # + 1 when Start STT and Stop STT, no change when submit question
        self.transcript = ""
        self.history_transcript = ""
        self.rag_instance = None
        self.response_type = "concise mode"  # Default response type

        self.history_length = 3  # Number of previous interactions to consider
        # self.init_ui()
        self.setLayout(init_ui(self))
        self.loaded_files  = {}   # record the files that already in the source file
        self.history_text = []  # history conversation awareness
    
    
        self.recognized_text_Signal_stt.connect(lambda message: append_recognition(message, self.Transcripts_area))
        
        self.recognized_text_Signal_dialog.connect(self.question_entry.setText)  # Connect signal to update question bar
        self.submitQuestionSignal.connect(self.submit_question)
        self.stt_running = False  # Flag indicating running STT to prevent multiple STT processes
        self.voice_inputting = False  # Flag indicating voice input is in progress
        self.reading_text = False  # Flag indicating text-to-speech is in progress
        

        self.video_path = ""  # Store the video path
        self.video_status = {
            "video_transcript_flag": False,
            "video_frame_flag": False,
            "video_last_transcript": "",
            "video_last_frame": ""
        }
        
        # Enter key 
        self.installEventFilter(self)  # Install the event filter to handle window close event
        self.enter_key_pressed = False  # Flag to check if Enter key is pressed
        
        #space key
        space_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        space_shortcut.activated.connect(self.toggle_stt_button.click)
        
        # C key for conversation
        c_shortcut = QShortcut(QKeySequence(Qt.Key.Key_C), self)
        c_shortcut.activated.connect(self.toggle_conv_button.click)
        
        # R key for read aloud
        read_aloud_shortcut = QShortcut(QKeySequence(Qt.Key.Key_R), self)
        read_aloud_shortcut.activated.connect(self.read_aloud_button.click)
        
        # A key for correct content
        correct_shortcut = QShortcut(QKeySequence(Qt.Key.Key_A), self)
        correct_shortcut.activated.connect(self.correct_button.click)
        
        # E key for exit
        exit_shortcut = QShortcut(QKeySequence(Qt.Key.Key_E), self)
        exit_shortcut.activated.connect(self.close)
        
        # D key for if need retrieval
        retrieval_shortcut = QShortcut(QKeySequence(Qt.Key.Key_D), self)
        retrieval_shortcut.activated.connect(self.need_retrival_button.click)
        
        # switch from concise and detail mode
        response_detail_shortcut = QShortcut(QKeySequence(Qt.Key.Key_T), self)
        response_detail_shortcut.activated.connect(self.detailed_response_button.click)  # Default to concise mode
        
        response_concise_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Y), self)
        response_concise_shortcut.activated.connect(self.concise_button.click)  # Default to concise mode
        
        # summary button 
        summary_shortcut = QShortcut(QKeySequence(Qt.Key.Key_S), self)
        summary_shortcut.activated.connect(self.summary_button.click)  # summary button
        
        # RAG similarity threshold                
        self.RAG_thresh = 0.5 
        
#################### Video RAG Management Functions ########################

    def VideoRAG_manage(self):
        
        file_dialog = QFileDialog()
        video_path, _ = file_dialog.getOpenFileName(self, "Select File", "./source", " mp4 Files (*.mp4);;All Files (*)")

        self.upload_thread = VideoUploadThread(video_path)
        # self.upload_thread.finished.connect(self.handle_upload_success)
        # self.upload_thread.error.connect(self.handle_upload_error)
        self.upload_thread.start()

           
#################### Video Summary and Video Parser Functions #######################           
    def video_summary(self):
        self.video_summary_thread = VideoSummaryThread(self.video_path)
        self.video_summary_thread.status_update.connect(lambda msg: print(msg))  
        self.video_summary_thread.summary_update.connect(lambda msg: append_Video("Overall Summary", msg, self.summary_area))  # Update the overall summary area
        
        self.video_summary_thread.start()

    def video_parser(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select File", "./source", "mp4 Files (*.mp4);;All Files (*)")
        self.video_path = "./source/" + os.path.splitext(os.path.basename(file_path))[0] + ".md"  # Store the video md for later use
        
        if file_path:
            try:
                destination_folder = "./source/"
                os.makedirs(destination_folder, exist_ok=True)
                file_name = os.path.basename(file_path)
                shutil.copy(file_path, destination_folder)
                append_system_message(f"File '{file_name}' uploaded successfully to '{destination_folder}'.", self.QA_area)
            except Exception as e:
                append_system_message(f"Error uploading file: {e}", self.QA_area)
        
        coordinate_list = upload_mp4_with_roi(file_path)
       
        # Process video key frames  
        self.video_thread = VideoProcessingThread(file_path,self.video_status,  coordinate_list)
        self.video_thread.status_update.connect(lambda msg: print(msg))  # Update the status in the console
        self.video_thread.description_update.connect(lambda msg: append_Video("Frame", msg, self.video_area))  # Update the video area with frame descriptions
        # self.video_thread.description_update.connect(lambda: setattr(self, 'video_frame_flag', True))  # update the last frame flag
        # self.video_thread.description_update.connect(lambda msg: setattr(self, 'video_last_frame', self.video_last_frame + msg))  # update the last frame
        self.video_thread.description_update.connect(lambda msg: self.video_status.update({
        "video_last_frame": self.video_status["video_last_frame"] + msg,
        "video_frame_flag": True}))
        self.video_thread.recent_summary_update.connect(lambda msg: append_Video("Recent Summary", msg, self.summary_area))  # Update the summary area with recent summaries
        self.video_thread.transcript_frame_finished.connect(lambda msg: self.video_status.update({
        "video_last_frame": '', "video_last_transcript": '',
        "video_frame_flag": False, "video_transcript_flag": False}))
        self.video_thread.description_update.connect(lambda: self.ask_llm())  # when update, refresh the RAG
        self.video_thread.finished_processing.connect(lambda path: print(f"Markdown saved: {path}"))
        self.video_thread.start()
                
        # Process the audio stream in the video and perform real-time transcription
        self.transcription_thread = RealTimeStreamingTranscriptionThread(file_path)
        # Connect the status update signal to a function that appends messages to the video area
        self.transcription_thread.status_update.connect(lambda msg: append_Video("Transcripts",msg, self.video_area))
        self.transcription_thread.transcript_update.connect(lambda msg: append_Video("Transcripts",msg, self.video_area))
        
        self.transcription_thread.transcript_update.connect(lambda msg: self.video_status.update({
        "video_last_transcript": self.video_status["video_last_transcript"] + msg,
        "video_transcript_flag": True}))
 
        self.transcription_thread.transcript_update.connect(lambda: self.ask_llm())
        self.transcription_thread.finished_processing.connect(lambda: print("Transcript finished."))
        self.transcription_thread.start()
               

 ################## upload the file to the RAG pipeline #####################
    def ask_llm(self):
        """Initialize the LLM process."""
        self.submit_question_button.setEnabled(True)
        self.question_entry.setEnabled(True)
        self.question_entry.clear()

        pdf_file_path = "./source/"
        os.makedirs(pdf_file_path, exist_ok=True)  # Ensure the directory exists

        # Include the files in the source directory
        all_filename = [
            os.path.join(pdf_file_path, file) for file in os.listdir(pdf_file_path) 
                if file.endswith((".pdf", ".md", ".txt"))  # Include PDF, Markdown, and Text files into RAG
        ]

        
        # Identify new files that haven't been loaded yet, new_files a dictionary with filename and last modified time
        new_files = {}
        
        for file in all_filename:
            mtime = os.path.getmtime(file)
            if file not in self.loaded_files or mtime > self.loaded_files[file]:
                new_files[file] = mtime
                self.loaded_files[file] = mtime  # Update the last modified time  
                  
        if not new_files:
            # Print message to the left side for no new files
            append_system_message("No new files to load into the LLM. Initializing LLM with default setup.", self.QA_area)

        try:
            # Initialize or reuse the RAG pipeline
            if self.rag_instance is None:
                self.rag_instance = RAG_pipeline()

            # Process the documents and build the QA chain
            docs = self.rag_instance.load_documents(new_files)
            
            # splits = self.rag_instance.split_documents(docs)
            splits = self.rag_instance.split_documents(docs)
            
            self.rag_instance.create_vector_db(splits, persist_directory='docs/chroma/')
            
            self.rag_instance.build_qa_chain(self.response_type)
            

            # Update loaded files and display success messages
            self.loaded_files.update(new_files)
                        
        except Exception as e:
            # Print error message to the left side
            append_system_message(f"Error loading transcription into LLM: {str(e)}", self.QA_area)


    # combine conversation history and user new question  (no need to append the history transcripts)
    def format_question_with_history(self, user_question, history_length=3):
        
        # Remove the last user question if it matches the current question
        history_without_latest = self.history_text[:-1] if self.history_text and self.history_text[-1].startswith("User question:") else self.history_text

        # Extract history_length most recent Q&A interactions
        history_context = "\n".join(history_without_latest[-history_length * 2:])  
        
        formatted_question = f"Previous interactions:\n{history_context}\nUser question: {user_question}"
        
        return formatted_question
    
    
    def submit_question(self):
        """Submit a question to the LLM and display the answer."""
        
        question = self.question_entry.text().strip()
            
        if not question:
            QMessageBox.warning(self, "Warning", "Please enter a question.")
            return
        
        # when running STT, want to ask question real time (trigger signal only : submit question)
        # 1. Save the history transcript into pdf file: named as transcription_chunk_{current_chunk_number}.pdf
        # 2. upload the pdf file to the RAG pipeline (ask_llm function)
        # 3. ask the LLM for answer
        if self.stt_running:
            if self.rag_instance is None:
                self.rag_instance = RAG_pipeline()
            save_transcription(self.history_transcript, self.current_chunk_number) # step 1
            docs = self.rag_instance.load_documents([f"./source/transcription_chunk_{self.current_chunk_number}.pdf"]) # step 2
            splits = self.rag_instance.split_documents(docs)
            self.rag_instance.create_vector_db(splits, persist_directory='docs/chroma/')    
            self.loaded_files.update({f"./source/transcription_chunk_{self.current_chunk_number}.pdf": os.path.getmtime(f"./source/transcription_chunk_{self.current_chunk_number}.pdf")})
        
        self.history_text.append(f"User question: {question}")  # Append the question to the history text
        
                
        if self.need_retrival_button.isChecked():
            # Use the RAG pipeline to generate an answer
            try:
                if not self.loaded_files:
                    # if no files loaded, directly get the answer from the LLM
                    if  self.response_type == "concise mode":
                        formatted_question = f"{self.format_question_with_history(user_question=question, history_length=self.history_length)}. Must give me a clear answer as concise as you can (no nessecary to be a sentence). First find answer in history transcription, if no include, then you answer freely."
    
                    elif self.response_type == "detail mode":
                        formatted_question = f"{self.format_question_with_history(user_question=question, history_length=self.history_length)} Give me a clear answer within 2 or 3 sentences. First find answer in history transcription, if no include, then you answer freely."
    
                    
                    answer = llm_direct_response(formatted_question)
                
                    self.history_text.append(f"Answer: {answer}")  # Append the answer to the history text

                    append_conversation(self.QA_area, question, answer)
                
                    print (f"formatted question is:\n{formatted_question}")
                    print("adaptive RAG not used, no files loaded.")
                
                # Display the question and answer

                    self.question_entry.clear()
                    
                # if files loaded, we use RAG to get the answer                    
                else:
                    # Adaptive RAG
                    # 1. get the score of the RAG first
                    if self.rag_instance is None:
                        self.rag_instance = RAG_pipeline()
                    rag_score = self.rag_instance.get_score(question, k=13)
                    print(f"only question Similarity Score (distance): {rag_score}\n")
                    print("="*10)
                    
                    # 2. if the score is lower than threshold, we use RAG (high similarity)
                    
                    if rag_score < self.RAG_thresh:
                        self.rag_instance.build_qa_chain(self.response_type)
                        
                        formatted_question = f"{self.format_question_with_history(user_question=question, history_length=self.history_length)}."
                        
                        answer, similarity_score, source_pdfs, Time_taken = self.rag_instance.generate_answer(formatted_question, k=13) if self.rag_instance else "[Mock Answer: LLM not initialized]"
                        
                        self.history_text.append(f"Answer: {answer}")  # Append the answer to the history text
                        
                        print (f"formatted question is: {formatted_question}")
                        
                        # Display the question and answer
                        append_conversation(self.QA_area, question, answer)
                        
                        print(f"formatted question Similarity Score(distance): {similarity_score}")
                        # print(f"Time taken: {Time_taken} seconds")
                        print("adaptive RAG used, retrieved.")
                                        
                        append_system_message(f"Source PDFs: {list(set(source_pdfs))}", self.QA_area)
                        
                        self.question_entry.clear()
                        
                    else:  # if the score is higher than the threshold, we dont use RAG (similarity low)
                        
                        if  self.response_type == "concise mode":
                            formatted_question = f"{self.format_question_with_history(user_question=question, history_length=self.history_length)}. Must give me a clear answer within 5 words (no nessecary to be a sentence). First find answer in history transcription, if no include, then you answer freely."
    
                        elif self.response_type == "detail mode":
                            formatted_question = f"{self.format_question_with_history(user_question=question, history_length=self.history_length)} Give me a clear answer within 2 or 3 sentences. First find answer in history transcription, if no include, then you answer freely."
    
                
                        answer = llm_direct_response(formatted_question)
                    
                        self.history_text.append(f"Answer: {answer}")  # Append the answer to the history text

                        append_conversation(self.QA_area, question, answer)
                    
                        print (f"formatted question is:\n{formatted_question}")
                        print ("adaptive RAG used, no retrieved.")
                        
                        
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error generating answer: {e}")
        
        # else:  # if need_retrieval_button is not checked, we directly get the answer from the LLM

        #     # Directly get the answer from the LLM
        #     try:
                
        #         if  self.response_type == "concise mode":
        #             formatted_question = f"{self.format_question_with_history(user_question=question, history_length=self.history_length)}. Must give me a clear answer within 5 words (no nessecary to be a sentence). First find answer in history transcription, if no include, then you answer freely."
    
        #         elif self.response_type == "detail mode":
        #             formatted_question = f"{self.format_question_with_history(user_question=question, history_length=self.history_length)} Give me a clear answer within 2 or 3 sentences. First find answer in history transcription, if no include, then you answer freely."
    
                
        #         answer = llm_direct_response(formatted_question)
                
        #         self.history_text.append(f"Answer: {answer}")  # Append the answer to the history text

        #         append_conversation(self.QA_area, question, answer)
                
        #         print (f"formatted question is:\n{formatted_question}")
                
        #         # Display the question and answer

        #         self.question_entry.clear()
                
        #     except Exception as e:
        #         QMessageBox.critical(self, "Error", f"Error generating answer: {e}")
        
        elif self.video_rag_answer_button.isChecked():
            try:
                
                formatted_question = f"{self.format_question_with_history(user_question=question, history_length=self.history_length)}. Must give me a clear answer within 5 words (no nessecary to be a sentence). First find answer in history transcription, if no include, then you answer freely."

                answer = get_videorag_answer(formatted_question)
                
                self.history_text.append(f"Answer: {answer}")  # Append the answer to the history text

                append_conversation(self.QA_area, question, answer)

                print (f"formatted question is:\n{formatted_question}")
                print("adaptive RAG not used, no files loaded.")
                

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error generating answer: {e}")

        if self.read_aloud_button.isChecked(): 
                speak_text_thread(answer)  # Call the text-to-speech function

    # Set the response type based on the selected radio button
    def set_response_type(self):
        """Set the response type based on the selected radio button."""
        if self.concise_button.isChecked():
            
                self.response_type = "concise mode"

        elif self.detailed_response_button.isChecked():
            
                self.response_type = "detail mode"
 
    # Text-to-Speech Function 
    def speak_text(self, text):
        """Convert text to speech and play it."""
        
        # Speak text and pause recognition while speaking.
        
        self.reading_text = True  # Set flag to True
        engine = pyttsx3.init()
        
        voices = engine.getProperty('voices')
        
        engine.setProperty('voice', voices[1].id)
        engine.setProperty("rate", 200)  # Adjust speed 
        engine.setProperty("volume", 1.0)  # Adjust volume
        engine.say(text)
        engine.runAndWait()
        
        self.reading_text = False  # Reset flag 
                
############# Voice query Input Functions #########################
    def start_voice_input(self):
        if self.enter_key_pressed:
            return
        
        self.voice_inputting = True  # Set flag to True
        self.enter_key_pressed = True  # Set flag to True
        stop_listening_event.clear()  # Reset the stop event for STT
        
        append_system_message("Listening for voice input...", self.Transcripts_area)
        self.voice_input_button.setText("Listening...")  # Change button text to indicate active listening
        self.voice_input_button.setStyleSheet("font-size:14px; background-color: red; color:white;")  # Change color
        
        stt_thread = threading.Thread(target=self.capture_voice_input, daemon=True)
        stt_thread.start()

        return
    
    # Voice query Input Functions
    def stop_voice_input(self):
        
        if not self.enter_key_pressed:
            return
        
        self.enter_key_pressed = False  # Reset flag
        self.voice_inputting = False  # Reset flag
        stop_listening_event.set()  # Stop STT process
        self.voice_input_button.setText("Voice Input")  # Reset button text
        self.voice_input_button.setStyleSheet("font-size:14px; background-color: lightgrey; color:black;")  # Reset color
        append_system_message("Voice input stopped. Processing...", self.Transcripts_area)
        self.submit_question()  # Submit the question after voice input is stopped
        return
    
    # Voice query Input Functions
    def capture_voice_input(self):
        """Capture voice input and set it in the input box."""
        transcript = stt_for_query()
        self.question_entry.setText(transcript)  # Set recognized text in the input field
        self.recognized_text_Signal_dialog.emit(transcript)   # Signal to update the question entry field
###### Voice query Input Functions finish#########################        
                
############## Start speech-to-text recognition #######################
    def start_stt(self):
        """Start the Speech-to-Text process."""
        self.need_retrival_button.setChecked(True)  # automatically set to need retrieval mode in start retrieval 
        append_system_message("Listening...", self.Transcripts_area)
        
        global stop_recognition_sep
        stop_recognition_sep.clear()  # Reset the stop event
        
        # Start STT in a separate thread
        stt_thread = threading.Thread(target=self.run_stt, daemon=True)
        stt_thread.start()

    # Stop speech-to-text recognition
    def stop_stt(self):
        
        """Stop the Speech-to-Text process."""
        global stop_recognition_sep
        stop_recognition_sep.set()
        print("Stopping STT...")
        self.history_transcript = ""  # Clear the history transcript cache
        
    
    def run_stt(self):
        """STT process."""
               
        if self.stt_running:  # Avoid multiple execution
            return
        self.stt_running = True  # Set flag to True

        try:
            
            def update_ui(recognized_text, channel_tag):
                channel_label = f"Ch{channel_tag}" # Default
                
                if channel_label == "Ch1":
                    channel_label = "User"
                elif channel_label == "Ch2":
                    channel_label = "Speaker"
                    
                display_text = f"[{channel_label}]: {recognized_text}"
                self.recognized_text_Signal_stt.emit(display_text)  # Emit the signal to update the UI
                self.history_transcript += display_text + '\n'
                
                if channel_tag == 2:
                    if self.correct_button.isChecked():
                        #1. directly use the LLM's knowledge
                        if self.loaded_files == {}:  # if no files loaded, we use the LLM's knowledge to correct the recognized text
                            correction = content_correct(recognized_text)  # Get correction from the LLM
                        #2. use the RAG pipeline to get the correction if there are files loaded
                        else:  # if files loaded, we use the RAG pipeline to get the correction
                            if self.rag_instance is None:
                                self.rag_instance = RAG_pipeline()
                            if self.rag_instance.build_qa_chain_correct() is None:
                                self.rag_instance.build_qa_chain_correct()
                            correction, similarity_score_correct,_ ,_ = self.rag_instance.generate_answer(recognized_text)
                        print(f"recognized text is:\n{recognized_text}\n")
                        print(f"correction is:\n{correction}\n")
                        if correction not in ["correct", "Correct", "Correct."]:
                            append_system_message(f"Warning: {correction}", self.Transcripts_area)
                            if self.read_aloud_button.isChecked():  # Read aloud the correction if the read aloud button is checked
                                speak_text_thread(correction)  # Read aloud the correction
                        
            # Call the STT function with the update callback
            
            # continuous_recognition_left(self.current_chunk_number, update_ui_left)
            # continuous_recognition_right(self.current_chunk_number, update_ui_right)
            continuous_recognition_sep(self.current_chunk_number, update_ui)
            
            append_system_message(f"Transcription saved as transcription_chunk_{self.current_chunk_number}.pdf", self.Transcripts_area)
            self.current_chunk_number += 1
        
        except Exception as e:
            append_system_message(f"Error during transcription: {str(e)}", self. _area)
        finally:
            
            self.stt_running = False  # Reset flag
            
            self.ask_llm()
            # Reset the toggle button state once STT processing is finished.
            self.toggle_stt_button.setChecked(False)
            self.toggle_stt_button.setText("Start STT")      
############## Stop speech-to-text recognition #######################

###### Dialog block for conversation start: ###################################################
    def start_conversation(self):
        """Start the conversation process."""
        print("Starting conversation...")
        global stop_recognition
        stop_recognition.clear()  # Reset the stop event
        
        speak_text_thread("Hello, how can I help you today?")
        append_system_message("Conversation started.", self.QA_area)
        self.read_aloud_button.setChecked(True)
        
        conversation_thread = threading.Thread(target=self.run_conversation, daemon=True)
        conversation_thread.start()
        

    def stop_conversation(self):
        """Stop the conversation process."""
        print("Stopping conversation...")
        
        global stop_recognition
        stop_recognition.set()
        
        
    def run_conversation(self):
        
        def update_question(recognized_text):
            
            user_text = recognized_text.strip()
            if not user_text:
                return
            if not self.reading_text:   # only submit question when the text-to-speech is not running
                print(f"Recognized text: {recognized_text}")
                self.recognized_text_Signal_dialog.emit(recognized_text)
                self.submitQuestionSignal.emit()   # Emit the signal to submit the question

        continuous_recognition(self.current_chunk_number, update_question)
####### Dialog block for conversation finish ###############################################   
 
    
#################### Upload file ######################
# If upload a pdf file, extract images        
    def upload_file(self):
        """Allow the user to upload a file to the project/source directory."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select File", "./source", "PDF, Word, PowerPoint, Excel, txt, PNG and mp4 Files (*.pdf *docx *xlsx *pptx *.png *.jpg *.jpeg *.txt *.mp4);;All Files (*)")

        if file_path:
            try:
                
                destination_folder = "./source/"
                os.makedirs(destination_folder, exist_ok=True)  # Ensure the directory exists
                file_name = os.path.basename(file_path)  # Get the file name from the path
                
                shutil.copy(file_path, destination_folder)  # Copy the file to the target directory

                # Print success message on the left side
                append_system_message(f"File '{file_name}' uploaded successfully to '{destination_folder}'.", self.QA_area)
            except Exception as e:
                # Print error message on the left side
                append_system_message(f"Error uploading file: {e}", self.QA_area)
        
        if file_path.endswith(".pdf"):
            extract_text_and_images(file_path)
            self.ask_llm()
            
        elif file_path.endswith((".png", ".jpg", ".jpeg")):
            Imagett.image_to_txt(file_path)
            self.ask_llm()
            
        elif file_path.endswith((".docx" , ".doc" , ".xlsx" , ".xls" , ".pptx" , ".ppt")):
            office_2_pdf.office_2_pdf(file_path)
            self.ask_llm()
            
        elif file_path.endswith(".mp4"):
            # coordinate_list = upload_mp4_with_roi(file_path)
            asyncio.run(process_video_combined_async(file_path))  # Process the video file asynchronously
            self.ask_llm()
            

    # Event filter to capture Enter key press for voice input
    def eventFilter(self, obj, event):
        """Detect Enter key press and release to trigger voice input."""
        # 1) Check for auto-repeat on KeyPress
        if event.type() == QEvent.Type.KeyPress:
            # IGNORE auto-repeat keypresses
            if event.isAutoRepeat():
                return True  # Stop here, do not trigger start_voice_input()

            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                # 2) Normal (non-repeated) keypress
                if not self.enter_key_pressed:
                    self.start_voice_input()
                return True  # Stop event propagation

        # 3) KeyRelease event
        elif event.type() == QEvent.Type.KeyRelease:
            # KeyRelease can also be autoRepeated, so check that too:
            if event.isAutoRepeat():
                return True  # Ignore auto-repeat KeyRelease

            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if self.enter_key_pressed:
                    self.stop_voice_input()
                return True  # Stop event propagation

        return super().eventFilter(obj, event)
    
    # Combine the space key press and release to trigger voice input
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.space_key_pressed = True
            self.start_voice_input()
        return super().keyPressEvent(event)
    
    

    # Toggle STT button function
    def toggle_stt(self):
        if self.toggle_stt_button.isChecked():
            # When checked, we want to start STT.
            self.toggle_stt_button.setText("Stop STT")
            self.start_stt()
        else:
            # When unchecked, stop STT.
            self.toggle_stt_button.setText("Start STT")
            self.stop_stt()
    
    # Toggle conversationa button function        
    def toggle_conversation(self):
        if self.toggle_conv_button.isChecked():
            # When checked, we want to start STT.
            self.toggle_conv_button.setText("Stop Conversation")
            if not self.stt_running:     # flag to check if STT is running (STT or conversation)
                self.start_conversation()
        else:
            # When unchecked, stop STT.
            self.toggle_conv_button.setText("Start Conversation")
            self.stop_conversation()


    def closeEvent(self, event):
        clear_source_directory()
        result = clean_server_workdir()
        if result is None:
            print("Server not connected, skip remote cleanup.")
        else:
            print("Remote server cleanup success:", result)
        event.accept()
                        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ConvoAid()
    window.show()
    sys.exit(app.exec())



