import sys
import os
import subprocess
import threading
import queue
import textwrap
import time
import glob

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QLabel,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import QThread, pyqtSignal, QTimer, Qt

# Google STT 
from google.cloud import speech
from google.oauth2 import service_account

# PDF saving
from fpdf import FPDF

# ============ Global ============
api_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "api")
CREDENTIAL_PATH = glob.glob(os.path.join(api_dir, "*.json"))[0]

if not CREDENTIAL_PATH:
    raise FileNotFoundError("No JSON credential file was found in the ./api directory.")



STOP_STREAM_AFTER_SECONDS = 240         
CHUNK_SIZE = 4096                       
SAMPLE_RATE = 16000                    

# ============ Threading ============
class RealTimeStreamingTranscriptionThread(QThread):   
    """
    Use ffmpeg to read audio from MP4 file, then use Google Cloud Speech-to-Text's
    'streaming_recognize' for real-time transcription, automatically segmenting to avoid
    """
    # Define the signals
    transcript_update = pyqtSignal(str)      # When new transcript is available
    status_update = pyqtSignal(str)          # When status updates are needed
    finished_processing = pyqtSignal()       # When processing is finished

    def __init__(self, mp4_file_path: str, parent=None):
        super().__init__(parent)
        self.mp4_file_path = mp4_file_path
        self._stop_flag = False  
        self.md_filename = os.path.basename(mp4_file_path).replace(".mp4", ".md")
        self.md_file_path = os.path.join("./source", self.md_filename)        
        
        # Credential path should be set in the environment variable or directly here
        self.credentials = service_account.Credentials.from_service_account_file(
            CREDENTIAL_PATH
        )
        self.client = speech.SpeechClient(credentials=self.credentials)

        # Outpu PDF file name
        self.chunk_index = 1
        self.global_transcript = ""  

        # Control chunk
        self.chunk_stop = threading.Event()

    def stop(self):
        """
        Call this method to stop the transcription thread gracefully.
        """
        self._stop_flag = True
        self.chunk_stop.set()

    def run(self):
        """
        QThread main function.
        """
        if not os.path.isfile(self.mp4_file_path):
            self.status_update.emit(f"Error: File not exit {self.mp4_file_path}")
            self.finished_processing.emit()
            return

        # Send initial status update
        self.status_update.emit("Start processing...")

        # Prepare the audio queue
        audio_queue = queue.Queue()

        # Read ffmpeg Audio in subprocess thread
        reader_thread = threading.Thread(
            target=self._read_from_ffmpeg,
            args=(self.mp4_file_path, audio_queue),
            daemon=True
        )
        reader_thread.start()


        # Do chunk recognition until the audio is fully read or manually stopped
        try:
            while not self._stop_flag:
                self.chunk_stop.clear()
                pdf_filename = f"transcription_chunk_{self.chunk_index}.pdf"

                # Start recognizing the current chunk
                self._recognize_chunk(audio_queue, pdf_filename)
                self.chunk_index += 1

            
                if self._stop_flag:
                    break

                
                if not reader_thread.is_alive() and audio_queue.empty():
                    self.status_update.emit("Audio reading finished.")
                    break

            # Wait for the reader thread to finish
            reader_thread.join()
        except Exception as e:
            self.status_update.emit(f"Error during transcript: {e}")
        finally:
            # Send finished_processing signal 
            self.status_update.emit("Processing finished.")
            self.finished_processing.emit()

    def _read_from_ffmpeg(self, mp4_file_path, audio_queue):
        """
        Call ffmpeg and convert mp4 to PCM , put into audio_queue。
        Unit files finish reading or stop_flag is True。
        """
        cmd = [
            "ffmpeg",
            "-i", mp4_file_path,
            "-f", "s16le",  # 
            "-acodec", "pcm_s16le",
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            "-vn",
            "-"  
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


        while not self._stop_flag:
            data = process.stdout.read(CHUNK_SIZE)
            if not data:
                break
            audio_queue.put(data)
            
            duration = CHUNK_SIZE / (SAMPLE_RATE * 2)  
            time.sleep(duration)

        process.stdout.close()
        process.wait()

        self._stop_flag = True  

    def _recognize_chunk(self, audio_queue, pdf_filename: str):
        """
        Do streaming recognition for a single "chunk" of audio, with a time limit set by STOP_STREAM_AFTER_SECONDS.
        """


        # Define a timer to automatically stop the current chunk recognition
        timer = QTimer()
        timer.setSingleShot(True)

        # Using lambda to set chunk_stop when the timer times out
        timer.timeout.connect(lambda: self.chunk_stop.set())
        timer.start(STOP_STREAM_AFTER_SECONDS * 1000)

        # streaming_config
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE,
            language_code="en-US",  # Set your language here
            enable_automatic_punctuation=True
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=True  #
        )

        local_transcript = ""

        def request_generator():
            """
            Generate streaming requests from the audio queue.
            """
            while not self.chunk_stop.is_set():
                try:
                    data = audio_queue.get(timeout=0.5)
                    yield speech.StreamingRecognizeRequest(audio_content=data)
                except queue.Empty:
                    # If the queue is empty, check if we should stop
                    if self._stop_flag:
                        break
                    continue

        # Start streaming_recognize
        responses = self.client.streaming_recognize(
            streaming_config,
            requests=request_generator()
        )

        # Process the responses
        try:
            for response in responses:
                if self.chunk_stop.is_set():
                    break
                for result in response.results:
                    if result.is_final:
                        recognized_text = result.alternatives[0].transcript
                        local_transcript += recognized_text + "\n"
                        
                        with open(self.md_file_path, "a", encoding = "utf-8") as md_file:
                            md_file.write("Transcript:")
                            md_file.write(recognized_text + "\n")
                        # Send a signal to update the main window
                        self.transcript_update.emit(recognized_text)
        except Exception as e:
            self.status_update.emit(f"[分段 {self.chunk_index}] 识别异常: {e}")
        finally:
            # Stop the timer
            timer.stop()

            with audio_queue.mutex:
                audio_queue.queue.clear()


