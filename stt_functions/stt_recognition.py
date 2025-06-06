import speech_recognition as sr
from google.cloud import speech
from google.oauth2 import service_account
from collections import defaultdict
import threading
import textwrap
from fpdf import FPDF
import wave
import io
import pyaudio
from pydub import AudioSegment
import time
import queue
import sys
import os
import glob

# Google Speech-to-Text credentials
api_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "api")
json_files = glob.glob(os.path.join(api_dir, "*.json"))

if not json_files:
    raise FileNotFoundError("No JSON credential file was found in the ./api directory.")

credentials = service_account.Credentials.from_service_account_file(json_files[0])

# # # initial Google Speech-to-Text client
# # client = speech.SpeechClient(credentials=credentials)

# # # initialize the recognizer
recognizer = sr.Recognizer()

# Word wrapping setup (wrap at 80 characters per line)
wrapper = textwrap.TextWrapper(width=80)

# Event to stop the recording
stop_listening_event = threading.Event()


##########################################################################################

def listen_and_recognize_multi(current_chunk_number, min_speaker_count = 1, max_speaker_count = 3):
    
        FORMAT = pyaudio.paInt16  # 
        CHANNELS = 1  #
        RATE = 48000  # sample rate
        CHUNK = 512  # buffer
        OUTPUT_FILENAME = f"./source/recording_{current_chunk_number}.wav"  # file name

        # Initialize the PDF document
        pdf_filename = f"transcription_chunk_{current_chunk_number}.pdf"
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # initial pyaudio
        audio = pyaudio.PyAudio()
        # Store audio chunks
        frames = []
        
        # Store the transcription in a buffer
        transcription = ''

        # 
        stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)

        print("Start Recording...")

        try:
            while not stop_listening_event.is_set():
                stream.start_stream()
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
        except Exception as e:
            print(f"Error during recording: {e}")

        # stop the stream, close it, and terminate the pyaudio instantiation
        stream.stop_stream()
        stream.close()
        audio.terminate()
        print("Recording stopped.")
        

        # save the audio data to wave file
        with wave.open(OUTPUT_FILENAME, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))

        print(f"Audio recorded and saved as {OUTPUT_FILENAME}")
        
        
        # read wave and recognize the speech
        client = speech.SpeechClient(credentials=credentials)
        
        with open(OUTPUT_FILENAME, "rb") as audio_file:
            content = audio_file.read()
        
        audio = speech.RecognitionAudio(content=content)  # content is the data of the audio file



        diarization_config = speech.SpeakerDiarizationConfig(
            enable_speaker_diarization=True,
            min_speaker_count=min_speaker_count,
            max_speaker_count=max_speaker_count,
        )

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=48000,
            language_code="en-US",
            diarization_config=diarization_config,
        )

        # send request

        response = client.recognize(config=config, audio=audio)
        
        
        speaker_transcripts = defaultdict(list)

        for result in response.results:
            words_info = result.alternatives[0].words
            for word_info in words_info:
                speaker_transcripts[word_info.speaker_tag].append(word_info.word)

        # Output the transcript for each speaker
        for speaker, transcript in speaker_transcripts.items():
            if speaker != 0:
                print(f"Speaker {speaker}: {' '.join(transcript)}")
                
                speaker_text = f"Speaker {speaker}:"
                transcription = ' '.join(transcript)
                
                # Wrap text for better PDF formatting
                wrapped_transcription = wrapper.fill(transcription)

                # Add speaker heading and transcription to the PDF
                pdf.multi_cell(0, 10, speaker_text)  # Add Speaker tag
                pdf.multi_cell(0, 10, wrapped_transcription)  # Add wrapped transcription
                pdf.ln()  # Add a blank line for separation

        
        # Save the transcription to a PDF once Enter is pressed
        pdf.output(f"./source/{pdf_filename}")
        print(f"Transcription saved to {pdf_filename}\n")
        


import audioop

RATE = 16000 # Recommended Rate
CHUNK_SIZE = int(RATE / 10) # 100ms chunks
CHANNELS = 2
FORMAT = pyaudio.paInt16
SAMPLE_WIDTH = pyaudio.get_sample_size(FORMAT)
# --- Silence Detection Parameters ---
SILENCE_LIMIT_SECONDS = 2.0 # How many seconds of silence triggers stop
SILENCE_THRESHOLD = 500     # Amplitude threshold - TUNE THIS VALUE
# --- End Silence Detection ---
# --- Queues and Globals ---
stereo_audio_queue = queue.Queue()
mono_audio_queue = queue.Queue()
final_transcript = ""
transcript_lock = threading.Lock()
# --- End Queues and Globals ---


def capture_audio(stop_event):
    """Captures stereo audio, puts chunks into queue, and stops on silence."""
    pa = pyaudio.PyAudio()
    stream = None
    print("Initializing audio stream for capture...")

    # Calculate number of silent chunks needed to trigger stop
    num_silent_chunks_needed = int(SILENCE_LIMIT_SECONDS * RATE / CHUNK_SIZE)
    silent_chunks_count = 0
    has_started_speaking = False # Flag to avoid stopping immediately if starts silent

    try:
        stream = pa.open(format=FORMAT,
                         channels=CHANNELS,
                         rate=RATE,
                         input=True,
                         frames_per_buffer=CHUNK_SIZE)

        print(f"Capture thread listening... (will stop after {SILENCE_LIMIT_SECONDS}s of silence)")

        while not stop_event.is_set():
             try:
                 data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                 stereo_audio_queue.put(data) # Put data in queue regardless of silence initially

                 # --- Silence Detection Logic ---
                 rms = audioop.rms(data, SAMPLE_WIDTH)
                 # print(f"RMS: {rms}") # Optional: Print RMS for tuning threshold

                 if rms < SILENCE_THRESHOLD:
                     if has_started_speaking: # Only count silence after speech started
                         silent_chunks_count += 1
                     #else: still silent before speech, do nothing special
                 else:
                     # Sound detected, reset silence counter and mark that speech has started
                     silent_chunks_count = 0
                     if not has_started_speaking:
                         print("Speech detected.")
                         has_started_speaking = True

                 # Check if silence limit is reached AFTER speech has started
                 if has_started_speaking and silent_chunks_count > num_silent_chunks_needed:
                     print(f"Silence detected for > {SILENCE_LIMIT_SECONDS} seconds. Stopping capture.")
                     stop_event.set() # Signal other threads to stop
                     # No need to break here, let the main stop_event check handle loop exit
                 # --- End Silence Detection Logic ---

             except IOError as ex:
                 print(f"Capture thread IOError: {ex}")
                 time.sleep(0.01)

        print("Capture thread stop condition met.")

    except Exception as e:
        print(f"Error in capture_audio thread: {e}")
        stop_event.set() # Ensure stop is signalled on error
    finally:
        if stream is not None:
            try:
                if stream.is_active(): stream.stop_stream()
                stream.close()
            except Exception as close_err: print(f"Error closing capture stream: {close_err}")
        if pa is not None: pa.terminate()
        stereo_audio_queue.put(None) # IMPORTANT: Always signal processing thread to stop
        print("Capture thread finished.")

# --- process_audio function remains the same ---
def process_audio(stop_event):
    """Processes stereo chunks from queue, extracts left channel, puts mono chunks into mono_audio_queue."""
    print("Processing thread started.")
    while True: # Removed 'not stop_event.is_set()' here, rely on sentinel
        stereo_data = stereo_audio_queue.get()
        if stereo_data is None: # Check for sentinel
            mono_audio_queue.put(None) # Pass sentinel along
            stereo_audio_queue.task_done() # Mark sentinel as done
            break

        # --- Add task_done() in a finally block for robustness ---
        try:
            stereo_segment = AudioSegment(
                data=stereo_data, sample_width=SAMPLE_WIDTH,
                frame_rate=RATE, channels=CHANNELS
            )
            # ... (rest of the pydub processing logic as before) ...
            if stereo_segment.channels == 2:
                mono_segments = stereo_segment.split_to_mono()
                left_channel_segment = mono_segments[0]
            else:
                left_channel_segment = stereo_segment

            buffer = io.BytesIO()
            left_channel_segment.export(buffer, format="wav")
            mono_chunk_data = buffer.getvalue()
            # --- End pydub processing ---

            mono_audio_queue.put(mono_chunk_data)

        except Exception as e:
            print(f"Error processing chunk: {e}")
            # Decide: skip chunk or stop everything? Continue for now.
        finally:
             # Ensure task_done is called even if processing fails for this chunk
             # But only if it wasn't the sentinel value we just broke on
             if stereo_data is not None:
                 stereo_audio_queue.task_done()


    print("Processing thread finished.")


# --- generate_google_requests function remains the same ---
def generate_google_requests():
    """Generator that yields StreamingRecognizeRequest objects from the mono queue."""
    print("Google request generator started.")
    while True:
        chunk = mono_audio_queue.get()
        if chunk is None: # Sentinel check
            mono_audio_queue.task_done() # Mark sentinel as done
            break
        # --- Add task_done() in finally block ---
        try:
             yield speech.StreamingRecognizeRequest(audio_content=chunk)
        finally:
             # Ensure task_done is called even if consumer stops early
             if chunk is not None:
                 mono_audio_queue.task_done()

    print("Google request generator finished.")


# --- stt_for_query_streaming function main logic remains mostly the same ---
# (It will now stop when the capture thread stops due to silence)
def stt_for_query():
    """
    Performs streaming STT using PyAudio, Pydub, and Google Cloud Speech API,
    with automatic stopping based on silence detection.
    """
    global final_transcript
    final_transcript = "" # Reset transcript
    stop_event = threading.Event()

    # Clear queues from previous runs if any
    while not stereo_audio_queue.empty(): stereo_audio_queue.get_nowait()
    while not mono_audio_queue.empty(): mono_audio_queue.get_nowait()


    # Start capture and processing threads
    capture_thread = threading.Thread(target=capture_audio, args=(stop_event,), daemon=True)
    process_thread = threading.Thread(target=process_audio, args=(stop_event,), daemon=True)

    capture_thread.start()
    time.sleep(0.1) # Give capture a tiny head start
    process_thread.start()

    # Configure Google Cloud Speech client and config
    client = speech.SpeechClient(credentials=credentials)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US",
        enable_automatic_punctuation=True,
        model="latest_long",
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True
    )

    print("Starting Google Cloud streaming recognition...")
    requests = generate_google_requests()
    responses = client.streaming_recognize(config=streaming_config, requests=requests)

    try:
        num_chars_printed = 0
        for response in responses:
             # ... (Interim/Final results printing logic remains the same) ...
            if not response.results: continue
            result = response.results[0]
            if not result.alternatives: continue
            transcript = result.alternatives[0].transcript
            overwrite_chars = ' ' * (num_chars_printed - len(transcript))
            if not result.is_final:
                sys.stdout.write("Interim: " + transcript + overwrite_chars + '\r')
                sys.stdout.flush()
                num_chars_printed = len(transcript)
            else:
                print("Final:   " + transcript + overwrite_chars)
                with transcript_lock:
                    final_transcript += transcript + " "
                num_chars_printed = 0


            # Check if the capture thread signalled stop (due to silence)
            # If so, we can break the response loop early.
            if stop_event.is_set():
                 print("Stop event detected in main loop, breaking response processing.")
                 break


    except Exception as e:
        print(f"Error during streaming recognition: {e}")
        # Ensure stop event is set if an error occurs in the main loop
        if not stop_event.is_set(): stop_event.set()
    finally:
        # --- Modified Finally Block (Removed Queue Joins) ---
        if not stop_event.is_set():
             print("Setting stop event in finally block...")
             stop_event.set() # Make sure it's set

        print("Waiting for threads to complete...")
        # Wait for threads to terminate
        capture_thread.join(timeout=3)
        process_thread.join(timeout=3)

        if capture_thread.is_alive(): print("Warning: Capture thread did not terminate.")
        if process_thread.is_alive(): print("Warning: Processing thread did not terminate.")
        # --- End Modified Finally Block ---

        print("Streaming finished.")
        with transcript_lock:
            return final_transcript.strip()

