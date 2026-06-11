"""Business logic ported from gui_beta.py (PyQt version).

Every function takes the shared AppState and EventBus instead of touching
widgets; UI updates that used to be signal/slot calls are now events
published on the bus and rendered by the React frontend.

Modules that require Google credentials (stt_functions.*) are imported
lazily inside the functions that need them, so the server can start and
serve the text-only QA flow even when ./api has no credential JSON.
"""

import os
import shutil
import threading

from RAG.rag import RAG_pipeline
from answer_request.llm_direct_response import llm_direct_response
from backend.files import save_transcription, clear_source_directory

SOURCE_DIR = "./source/"


# ---------------------------------------------------------------- events

def system_msg(bus, panel, text):
    """Equivalent of append_system_message(...) on a QTextEdit."""
    bus.publish({"type": "system", "panel": panel, "text": str(text)})


def publish_settings(state, bus):
    """Keep every connected client in sync with the server-side toggles."""
    bus.publish({"type": "settings", "settings": state.settings_dict(), "replay": False})


# ---------------------------------------------------------------- TTS

def speak(state, bus, text):
    """Synthesize speech server-side, play it in the browser (backend.tts)."""
    from backend import tts
    tts.speak(state, bus, text)


# ---------------------------------------------------------------- RAG

def refresh_rag(state, bus):
    """Port of ConvoAid.ask_llm(): ingest new ./source files into the RAG."""
    os.makedirs(SOURCE_DIR, exist_ok=True)

    all_filename = [
        os.path.join(SOURCE_DIR, file) for file in os.listdir(SOURCE_DIR)
        if file.endswith((".pdf", ".md", ".txt"))  # PDF, Markdown and Text go into the RAG
    ]

    with state.rag_lock:
        new_files = {}
        for file in all_filename:
            mtime = os.path.getmtime(file)
            if file not in state.loaded_files or mtime > state.loaded_files[file]:
                new_files[file] = mtime
                state.loaded_files[file] = mtime

        if not new_files:
            system_msg(bus, "qa", "No new files to load into the LLM. Initializing LLM with default setup.")

        try:
            if state.rag_instance is None:
                state.rag_instance = RAG_pipeline()

            docs = state.rag_instance.load_documents(new_files)
            # the Qt version fed an empty list into Chroma here, which raised
            # "Expected Embeddings to be non-empty" after every idle STT stop
            if docs:
                splits = state.rag_instance.split_documents(docs)
                state.rag_instance.create_vector_db(splits, persist_directory='docs/chroma/')
            if state.rag_instance.vector_db is not None:
                state.rag_instance.build_qa_chain(state.response_type)

            state.loaded_files.update(new_files)
        except Exception as e:
            system_msg(bus, "qa", f"Error loading transcription into LLM: {str(e)}")


def format_question_with_history(state, user_question, history_length=3):
    """Combine conversation history and the new user question."""
    history_without_latest = (
        state.history_text[:-1]
        if state.history_text and state.history_text[-1].startswith("User question:")
        else state.history_text
    )
    history_context = "\n".join(history_without_latest[-history_length * 2:])
    return f"Previous interactions:\n{history_context}\nUser question: {user_question}"


def _direct_answer(state, question):
    """Direct LLM answer with the concise/detail prompt wrapping."""
    formatted = format_question_with_history(state, user_question=question,
                                             history_length=state.history_length)
    if state.response_type == "concise mode":
        formatted_question = f"{formatted}. Must give me a clear answer within 5 words (no nessecary to be a sentence). First find answer in history transcription, if no include, then you answer freely."
    else:  # detail mode
        formatted_question = f"{formatted} Give me a clear answer within 2 or 3 sentences. First find answer in history transcription, if no include, then you answer freely."

    answer = llm_direct_response(formatted_question)
    print(f"formatted question is:\n{formatted_question}")
    return answer


def answer_question(state, bus, question):
    """Port of ConvoAid.submit_question(). Returns the answer text."""
    question = (question or "").strip()
    if not question:
        raise ValueError("Please enter a question.")

    # While STT runs, snapshot the live transcript into a PDF and ingest it
    # so the answer can use what was just said.
    if state.stt_running:
        with state.rag_lock:
            if state.rag_instance is None:
                state.rag_instance = RAG_pipeline()
            save_transcription(state.history_transcript, state.current_chunk_number)
            chunk_pdf = f"./source/transcription_chunk_{state.current_chunk_number}.pdf"
            docs = state.rag_instance.load_documents([chunk_pdf])
            splits = state.rag_instance.split_documents(docs)
            state.rag_instance.create_vector_db(splits, persist_directory='docs/chroma/')
            state.loaded_files.update({chunk_pdf: os.path.getmtime(chunk_pdf)})

    state.history_text.append(f"User question: {question}")

    source_pdfs = None
    if state.use_retrieval and state.loaded_files:
        # Adaptive RAG: retrieve only when the question is close enough to the corpus
        with state.rag_lock:
            if state.rag_instance is None:
                state.rag_instance = RAG_pipeline()
            rag_score = state.rag_instance.get_score(question, k=13)
            print(f"only question Similarity Score (distance): {rag_score}\n")
            print("=" * 10)

            if rag_score < state.RAG_thresh:
                state.rag_instance.build_qa_chain(state.response_type)
                formatted_question = f"{format_question_with_history(state, user_question=question, history_length=state.history_length)}."
                answer, similarity_score, source_pdfs, _time_taken = state.rag_instance.generate_answer(formatted_question, k=13)
                print(f"formatted question is: {formatted_question}")
                print(f"formatted question Similarity Score(distance): {similarity_score}")
                print("adaptive RAG used, retrieved.")
            else:
                answer = _direct_answer(state, question)
                print("adaptive RAG used, no retrieved.")
    else:
        answer = _direct_answer(state, question)
        if state.use_retrieval:
            print("adaptive RAG not used, no files loaded.")

    state.history_text.append(f"Answer: {answer}")
    bus.publish({"type": "qa", "question": question, "answer": str(answer)})
    if source_pdfs:
        system_msg(bus, "qa", f"Source PDFs: {list(set(source_pdfs))}")

    if state.read_aloud:
        speak(state, bus, answer)

    return answer


# ---------------------------------------------------------------- transcript routing

def handle_transcript_line(state, bus, recognized_text, channel_tag):
    """Port of the update_ui callback of ConvoAid.run_stt(): label a line,
    append it to the live transcript and run content correction for the
    Speaker channel. Audio now arrives from the browser (backend/audio.py)."""
    from stt_functions.content_correct import content_correct

    channel_label = f"Ch{channel_tag}"
    if channel_tag == 1:
        channel_label = "User"
    elif channel_tag == 2:
        channel_label = "Speaker"

    display_text = f"[{channel_label}]: {recognized_text}"
    bus.publish({"type": "transcript", "label": channel_label, "text": recognized_text})
    state.history_transcript += display_text + '\n'

    if channel_tag == 2 and state.correct_content:
        try:
            # 1. no files loaded -> correct with the LLM's own knowledge
            if state.loaded_files == {}:
                correction = content_correct(recognized_text)
            # 2. files loaded -> correct through the RAG pipeline
            else:
                with state.rag_lock:
                    if state.rag_instance is None:
                        state.rag_instance = RAG_pipeline()
                    if state.rag_instance.build_qa_chain_correct() is None:
                        state.rag_instance.build_qa_chain_correct()
                    correction, _score, _, _ = state.rag_instance.generate_answer(recognized_text)
            print(f"recognized text is:\n{recognized_text}\n")
            print(f"correction is:\n{correction}\n")
            if correction not in ["correct", "Correct", "Correct."]:
                system_msg(bus, "transcripts", f"Warning: {correction}")
                if state.read_aloud:
                    speak(state, bus, correction)
        except Exception as e:
            system_msg(bus, "transcripts", f"Correction error: {e}")


# ---------------------------------------------------------------- file upload

def save_upload(file_name, data):
    """Persist an uploaded file into ./source and return its path."""
    os.makedirs(SOURCE_DIR, exist_ok=True)
    file_name = os.path.basename(file_name)
    destination = os.path.join(SOURCE_DIR, file_name)
    with open(destination, "wb") as f:
        f.write(data)
    return destination


def process_upload(state, bus, file_path):
    """Port of ConvoAid.upload_file() conversion dispatch (runs in a thread)."""
    import file_conversion.Imagett as Imagett
    import file_conversion.office_2_pdf as office_2_pdf
    from file_conversion.pdf_split import extract_text_and_images

    file_name = os.path.basename(file_path)
    try:
        system_msg(bus, "qa", f"File '{file_name}' uploaded successfully to '{SOURCE_DIR}'.")

        if file_path.endswith(".pdf"):
            extract_text_and_images(file_path)
            refresh_rag(state, bus)
        elif file_path.endswith((".png", ".jpg", ".jpeg")):
            Imagett.image_to_txt(file_path)
            refresh_rag(state, bus)
        elif file_path.endswith((".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt")):
            office_2_pdf.office_2_pdf(file_path)
            refresh_rag(state, bus)
        elif file_path.endswith((".txt", ".md")):
            refresh_rag(state, bus)
    except Exception as e:
        system_msg(bus, "qa", f"Error uploading file: {e}")


def process_video_offline(state, bus, file_path, roi):
    """mp4 uploaded through File Management: non-realtime processing."""
    import file_conversion.video_process as video_process
    try:
        video_process.video_transcripts_frames(file_path, roi)
        refresh_rag(state, bus)
        system_msg(bus, "qa", f"Video '{os.path.basename(file_path)}' processed.")
    except Exception as e:
        system_msg(bus, "qa", f"Error processing video: {e}")


# ---------------------------------------------------------------- video parser (realtime)

def start_video_parse(state, bus, file_path, roi):
    """Port of ConvoAid.video_parser() minus the Qt file/ROI dialogs."""
    from backend.workers import VideoFrameWorker, VideoTranscriptionWorker

    state.video_path = "./source/" + os.path.splitext(os.path.basename(file_path))[0] + ".md"

    def on_description(msg):
        bus.publish({"type": "video", "head": "Frame", "text": msg})
        state.video_status.update({
            "video_last_frame": state.video_status["video_last_frame"] + msg,
            "video_frame_flag": True,
        })
        refresh_rag(state, bus)  # when update, refresh the RAG

    def on_recent_summary(msg):
        bus.publish({"type": "summary", "head": "Recent Summary", "text": msg})

    def on_transcript_frame_finished(_flag):
        state.video_status.update({
            "video_last_frame": '', "video_last_transcript": '',
            "video_frame_flag": False, "video_transcript_flag": False,
        })

    frame_worker = VideoFrameWorker(
        file_path, state.video_status, roi,
        on_status=lambda msg: print(msg),
        on_description=on_description,
        on_recent_summary=on_recent_summary,
        on_transcript_frame_finished=on_transcript_frame_finished,
        on_finished=lambda path: print(f"Markdown saved: {path}"),
    )
    frame_worker.start()

    def on_transcript(msg):
        bus.publish({"type": "video", "head": "Transcripts", "text": msg})
        state.video_status.update({
            "video_last_transcript": state.video_status["video_last_transcript"] + msg,
            "video_transcript_flag": True,
        })
        refresh_rag(state, bus)

    transcription_worker = VideoTranscriptionWorker(
        file_path,
        on_status=lambda msg: bus.publish({"type": "video", "head": "Transcripts", "text": msg}),
        on_transcript=on_transcript,
        on_finished=lambda: print("Transcript finished."),
    )
    transcription_worker.start()


def start_video_summary(state, bus):
    """Port of ConvoAid.video_summary()."""
    from backend.workers import run_video_summary
    threading.Thread(
        target=run_video_summary,
        args=(state.video_path,),
        kwargs={
            "on_status": lambda msg: print(msg),
            "on_summary": lambda msg: bus.publish({"type": "summary", "head": "Overall Summary", "text": msg}),
        },
        daemon=True,
    ).start()


# ---------------------------------------------------------------- shutdown

def clear_workspace():
    """Port of closeEvent(): wipe ./source and ./docs."""
    clear_source_directory()
