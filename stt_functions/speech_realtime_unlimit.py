import pyaudio
import queue
import sys
from google.cloud import speech
from google.oauth2 import service_account
import threading
from fpdf import FPDF
import textwrap
import time
import os
import glob

api_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "api")
json_files = glob.glob(os.path.join(api_dir, "*.json"))

if not json_files:
    raise FileNotFoundError("No JSON credential file was found in the ./api directory.")

credential = service_account.Credentials.from_service_account_file(json_files[0])

# Audio parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms
STOP_STREAM_AFTER_SECONDS = 240  # duration of each chunk in seconds

global_transcription = ''
transcription_lock = threading.Lock()
audio_queue = queue.Queue()

stop_recognition = threading.Event()
chunk_stop = threading.Event()

def callback(in_data, frame_count, time_info, status):
    """audio callback function, put audio data into queue"""
    if stop_recognition.is_set():
        return None, pyaudio.paComplete
    audio_queue.put(in_data)
    return None, pyaudio.paContinue

def continuous_recognition(current_chunk_number, update_callback=lambda x: None):
    """Continuous recognition with auto-restart"""
    global global_transcription
    internal_chunk_number = 0

    with transcription_lock:
        global_transcription = ''

    try:
        while not stop_recognition.is_set():
            print(f"\n=== start the {internal_chunk_number} chunk  (every {STOP_STREAM_AFTER_SECONDS} seconds) ===")

            recogn_thread = threading.Thread(
                target=recog_stream,
                args=(current_chunk_number, update_callback)
            )

            timer = threading.Timer(
                STOP_STREAM_AFTER_SECONDS,
                lambda: chunk_stop.set()
            )

            chunk_stop.clear()
            recogn_thread.start()
            timer.start()

            recogn_thread.join()
            timer.cancel()

            with audio_queue.mutex:
                audio_queue.queue.clear()

            internal_chunk_number += 1

    except KeyboardInterrupt:
        print("\nUser request stop...")
        stop_recognition.set()
        recogn_thread.join(timeout=1)
        timer.cancel()

def recog_stream(current_chunk_number, update_callback):
    """Google Cloud Speech-to-Text streaming"""
    pdf_filename = f"transcription_chunk_{current_chunk_number}.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Conversation -  {current_chunk_number}", ln=True, align="C")

    global global_transcription
    global chunk_stop

    client = speech.SpeechClient(credentials=credential)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US",
        enable_automatic_punctuation=True,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
    )

    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        stream_callback=callback
    )

    result_queue = queue.Queue()

    def generate_audio():
        while not chunk_stop.is_set() and not stop_recognition.is_set():
            try:
                data = audio_queue.get(timeout=1)
                yield speech.StreamingRecognizeRequest(audio_content=data)
            except queue.Empty:
                continue

    print("Listening... start speaking!")

    responses = client.streaming_recognize(streaming_config, generate_audio())

    try:
        for response in responses:
            for result in response.results:
                if result.is_final:
                    recognized_text = result.alternatives[0].transcript
                    with transcription_lock:
                        global_transcription += recognized_text + '\n'
                    update_callback(recognized_text)
            if chunk_stop.is_set():
                break

    except Exception as e:
        print("Error:", e)

    finally:
        print("Stop real-time recognition.")
        stream.stop_stream()
        stream.close()
        audio.terminate()

        if global_transcription.strip():
            wrapped_text = textwrap.fill(global_transcription, width=80)
            pdf.multi_cell(0, 10, wrapped_text)
            pdf.output(f"./source/{pdf_filename}")
            print(f"Conversation saved: {pdf_filename}")
        else:
            print("No transcription detected, skipping PDF save.")

    return result_queue