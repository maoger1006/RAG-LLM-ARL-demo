import pyaudio
import queue
import sys
from google.cloud import speech
from google.oauth2 import service_account
import threading
from fpdf import FPDF
import textwrap
import time
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from typing import Callable, Tuple, Dict
import os # Added for directory creation
import glob

# --- Configuration ---


api_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "api")
json_files = glob.glob(os.path.join(api_dir, "*.json"))

if not json_files:
    raise FileNotFoundError("No JSON credential file was found in the ./api directory.")

credential = service_account.Credentials.from_service_account_file(json_files[0])

OUTPUT_DIR = "./source/"

# Audio parameters
RATE = 16000  # Sample rate (Hz) - Ensure your mic supports this for stereo
CHUNK_DURATION_MS = 100 # Duration of each audio chunk in ms
CHUNK = int(RATE * CHUNK_DURATION_MS / 1000) # Samples per chunk
FORMAT = pyaudio.paInt16 # Sample format
CHANNELS = 2 # Capture stereo
SAMPLE_WIDTH = pyaudio.get_sample_size(FORMAT) # Bytes per sample (should be 2 for paInt16)

STOP_STREAM_AFTER_SECONDS = 240  # Duration of each recognition chunk before restart

# --- Global State & Synchronization ---
os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure output directory exists

# Queues
audio_queue = queue.Queue() # Raw stereo audio from PyAudio callback
left_channel_queue = queue.Queue() # Mono audio data for left channel
right_channel_queue = queue.Queue() # Mono audio data for right channel

# Stop Events
stop_recognition_sep = threading.Event() # Global stop signal for the entire process
stop_current_chunk = threading.Event()      # Stop signal for the current chunk processing

# Transcription storage (managed per chunk in the main loop)
# Using a lock if the update_callback modifies shared state directly,
# but preferred approach is to collect results in the main loop after threads join.
transcription_lock = threading.Lock()

# --- PyAudio Callback ---
def callback(in_data, frame_count, time_info, status):
    """audio callback function, put raw STEREO audio data into queue"""
    if stop_recognition_sep.is_set():
        return None, pyaudio.paComplete
    if in_data:
        audio_queue.put(in_data)
    return None, pyaudio.paContinue

# --- Audio Splitting Thread ---
def split_audio_loop():
    """
    Reads stereo chunks from audio_queue, splits them into mono,
    and puts them into left_channel_queue and right_channel_queue.
    """
    print("Audio splitter thread started.")
    while not stop_recognition_sep.is_set() and not stop_current_chunk.is_set():
        try:
            stereo_chunk_data = audio_queue.get(timeout=0.1)
            if stereo_chunk_data is None: # Check for potential sentinel value if needed
                continue

            # Create Pydub segment from raw stereo data
            try:
                stereo_segment = AudioSegment(
                    data=stereo_chunk_data,
                    sample_width=SAMPLE_WIDTH,
                    frame_rate=RATE,
                    channels=CHANNELS
                )
                # Split into two mono segments
                mono_channels = stereo_segment.split_to_mono()
                if len(mono_channels) == 2:
                    left_channel_queue.put(mono_channels[0].raw_data)
                    right_channel_queue.put(mono_channels[1].raw_data)
                else:
                    print(f"Warning: Pydub split did not return 2 channels (returned {len(mono_channels)}). Skipping chunk.")

            except CouldntDecodeError as pydub_err:
                print(f"Error decoding audio chunk with Pydub: {pydub_err}. Skipping chunk.")
            except Exception as e:
                 print(f"Error during audio splitting: {e}. Skipping chunk.")
            finally:
                 # Mark task as done regardless of success/failure in processing
                 # Avoids blocking if main thread joins the queue later
                 try:
                     audio_queue.task_done()
                 except ValueError: # Can happen if already marked done or queue cleared
                     pass


        except queue.Empty:
            # Queue is empty, just continue loop until stop is signaled
            if stop_recognition_sep.is_set() or stop_current_chunk.is_set():
                break
            continue
        except Exception as e:
            print(f"Error getting data from audio_queue in splitter: {e}")
            # Ensure task done even on unexpected errors retrieving from queue
            try:
                audio_queue.task_done()
            except ValueError:
                pass
            time.sleep(0.01) # Avoid busy-waiting on error

    print("Audio splitter thread finished.")

# --- Google STT Processing Thread (for one channel) ---
def process_channel_stream(
    channel_id: int,
    channel_queue: queue.Queue,
    current_chunk_number: int,
    update_callback: Callable[[str, int], None],
    results_container: Dict[int, str] # Dictionary to store final text for this chunk
):
    """
    Google Cloud Speech-to-Text streaming for ONE MONO channel.
    Reads mono audio data from its dedicated queue.
    Calls update_callback with final text and channel_id.
    Stores final text in results_container.
    """
    print(f"STT Processor started for Channel {channel_id} (Chunk {current_chunk_number}).")
    client = speech.SpeechClient(credentials=credential)
    # *** MONO RecognitionConfig ***
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US",
        enable_automatic_punctuation=True,
        audio_channel_count=1,  # Specify ONE channel
        # enable_separate_recognition_per_channel=False # This is default/not needed
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True # Keep interim if desired for UI feedback
    )

    def generate_audio_requests():
        """Generator yielding MONO audio chunks for this channel's Google API stream."""
        print(f"[{channel_id}] Audio chunk generator started...")
        sent_chunk_count = 0
        while not stop_recognition_sep.is_set() and not stop_current_chunk.is_set():
            try:
                mono_chunk_data = channel_queue.get(timeout=0.1)
                if mono_chunk_data:
                    yield speech.StreamingRecognizeRequest(audio_content=mono_chunk_data)
                    sent_chunk_count += 1
                    channel_queue.task_done() # Mark as done

            except queue.Empty:
                if stop_recognition_sep.is_set() or stop_current_chunk.is_set():
                    break
                continue
            except Exception as e:
                print(f"[{channel_id}] Error in generator loop: {e}")
                if 'mono_chunk_data' in locals() and mono_chunk_data:
                    try: channel_queue.task_done()
                    except ValueError: pass
                time.sleep(0.01)
        print(f"[{channel_id}] Audio chunk generator finished. Sent {sent_chunk_count} mono chunks.")

    requests = generate_audio_requests()
    try:
        responses = client.streaming_recognize(streaming_config, requests)

        # Process responses
        for response in responses:
            if stop_recognition_sep.is_set() or stop_current_chunk.is_set():
                print(f"[{channel_id}] Stop detected during response processing.")
                break

            if not response.results: continue

            result = response.results[0] # Only one result expected for mono
            if not result.alternatives: continue

            transcript = result.alternatives[0].transcript

            if not result.is_final:
                 # Optional: Handle interim results (e.g., display)
                 # sys.stdout.write(f"\rInterim Ch{channel_id}: {transcript}...")
                 # sys.stdout.flush()
                 pass
            else:
                final_text = transcript.strip()
                if final_text: # Only process if there's actual text
                    print(f"Chunk {current_chunk_number} Final Ch{channel_id}: {final_text}")

                    # Store final text for PDF generation later
                    with transcription_lock:
                         results_container[channel_id] += final_text + " " # Add space between utterances

                    # *** CALL EXTERNAL CALLBACK WITH CHANNEL INFO ***
                    update_callback(final_text, channel_id)

                    # Optional: Clear interim line if used
                    # sys.stdout.write("\r" + " " * len(f"Interim Ch{channel_id}: {transcript}...") + "\r")
                    # sys.stdout.flush()


    except StopIteration:
         print(f"[{channel_id}] Response stream ended normally.")
    except Exception as e:
        print(f"[{channel_id}] Error processing Google responses: {e}")
        # Handle potential gRPC errors like OutOfRange (often means end of audio/timeout)
        # or other API errors.
        if "OutOfRange" in str(e):
             print(f"[{channel_id}] Stream likely timed out or ended.")
        else:
             # Log other unexpected errors
             pass

    finally:
        # No stream/audio objects to close here as PyAudio is managed outside
        print(f"STT Processor finished for Channel {channel_id} (Chunk {current_chunk_number}).")


# --- Main Continuous Recognition Loop ---
def continuous_recognition_sep(initial_chunk_number=1, update_callback = lambda text, channel: None):
    """
    Manages the continuous recognition process:
    - Opens PyAudio stream.
    - Starts audio splitter thread.
    - Starts two STT processor threads (left & right channels).
    - Restarts processing chunk by chunk based on timer.
    - Collects results and saves PDF per chunk.
    """
    global stop_recognition_sep, stop_current_chunk

    internal_chunk_number = initial_chunk_number
    audio = None
    stream = None

    try:
        # --- Setup PyAudio ---
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS, # CAPTURE STEREO
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            stream_callback=callback # Puts raw stereo chunks in audio_queue
        )
        print("PyAudio Stereo Stream Opened.")
        stream.start_stream() # Start capturing audio

        while not stop_recognition_sep.is_set():
            print(f"\n=== Starting Recognition Chunk {internal_chunk_number} (max {STOP_STREAM_AFTER_SECONDS} seconds) ===")
            stop_current_chunk.clear()

            # --- Chunk-specific data ---
            # Dictionary to hold final transcriptions for this chunk from both channels
            chunk_transcriptions = {1: "", 2: ""}

            # --- Start Worker Threads for this Chunk ---
            splitter_thread = threading.Thread(target=split_audio_loop, daemon=True)

            left_stt_thread = threading.Thread(
                target=process_channel_stream,
                args=(1, left_channel_queue, internal_chunk_number, update_callback, chunk_transcriptions),
                daemon=True
            )
            right_stt_thread = threading.Thread(
                target=process_channel_stream,
                 args=(2, right_channel_queue, internal_chunk_number, update_callback, chunk_transcriptions),
                 daemon=True
            )

            splitter_thread.start()
            left_stt_thread.start()
            right_stt_thread.start()

            # --- Timer to stop the current chunk ---
            chunk_timer = threading.Timer(STOP_STREAM_AFTER_SECONDS, lambda: stop_current_chunk.set())
            chunk_timer.start()

            # --- Wait for chunk to finish (timer or global stop) ---
            # We wait on the timer/event, not directly joining threads yet,
            # as they might finish early if no speech is detected.
            while not stop_current_chunk.is_set() and not stop_recognition_sep.is_set():
                 # Check if worker threads are still alive - they might exit if stream ends
                 if not splitter_thread.is_alive() and not left_stt_thread.is_alive() and not right_stt_thread.is_alive():
                     print(f"Worker threads for chunk {internal_chunk_number} exited early.")
                     # Avoid timer cancelling if already stopped
                     if chunk_timer.is_alive():
                        chunk_timer.cancel()
                     stop_current_chunk.set() # Ensure the loop condition breaks
                     break
                 time.sleep(0.1) # Small sleep to avoid busy-waiting

            print(f"Signaling stop for chunk {internal_chunk_number}...")
            if chunk_timer.is_alive(): # Ensure timer is cancelled if we exited loop early
                chunk_timer.cancel()
            stop_current_chunk.set() # Explicitly set flag if not already set

            # --- Join Threads ---
            print("Waiting for threads to finish...")
            # It's important to join threads AFTER signaling them to stop.
            # Add timeouts to join to prevent indefinite hangs if a thread is stuck.
            join_timeout = 5.0 # seconds
            splitter_thread.join(timeout=join_timeout)
            left_stt_thread.join(timeout=join_timeout)
            right_stt_thread.join(timeout=join_timeout)
            print("Threads joined.")

            # Check if threads timed out during join
            if splitter_thread.is_alive(): print("Warning: Splitter thread did not exit cleanly.")
            if left_stt_thread.is_alive(): print("Warning: Left STT thread did not exit cleanly.")
            if right_stt_thread.is_alive(): print("Warning: Right STT thread did not exit cleanly.")


            # --- Clear Queues ---
            # Crucial to prevent data leaking between chunks
            print("Clearing residual queue data...")
            queues_to_clear = [audio_queue, left_channel_queue, right_channel_queue]
            for q in queues_to_clear:
                with q.mutex:
                    q.queue.clear()
                 # Reset task_done count if using queue.join() elsewhere, though not strictly necessary here
                 # with q.all_tasks_done: q.unfinished_tasks = 0


            # --- Process and Save PDF for the completed chunk ---
            print(f"Processing transcription for PDF (Chunk {internal_chunk_number})...")
            pdf_filename = f"transcription_chunk_{internal_chunk_number}.pdf"
            pdf = FPDF()
            pdf.add_page()
             # Add font that supports more characters if needed (e.g., DejaVu)
             # try:
             #    pdf.add_font('DejaVu', '', 'DejaVuSansCondensed.ttf', uni=True)
             #    pdf.set_font('DejaVu', size=12)
             # except RuntimeError:
             #     print("DejaVu font not found, using Arial (may cause encoding issues).")
            pdf.set_font("Arial", size=12) # Fallback font

            pdf.cell(0, 10, f"Conversation - Chunk {internal_chunk_number}", ln=True, align="C")

            has_content = False
            pdf_text = ""
            # Access the results collected during the chunk processing
            # Need lock here if threads were modifying this directly, but they modify their own copy passed in args
            # which is safer.
            # No lock needed here as we access chunk_transcriptions after threads have joined.
            left_text = chunk_transcriptions[1].strip()
            right_text = chunk_transcriptions[2].strip()

            # Simple interleaving - could be improved with timestamps if available
            # For now, just list all of Left then all of Right
            if left_text:
                pdf_text += "User:\n" + left_text + "\n\n"
                has_content = True
            if right_text:
                pdf_text += "Speaker:\n" + right_text + "\n\n"
                has_content = True

            if has_content:
                print(f"Wrapping text for PDF chunk {internal_chunk_number}...")
                # Encode carefully for PDF compatibility
                try:
                    # Attempt to encode relevant parts or replace problematic characters
                    # latin-1 is very limited; utf-8 is better but needs font support in FPDF
                    # Using replace for simplicity here
                    encoded_text = pdf_text.encode('latin-1', 'replace').decode('latin-1')
                    wrapped_text = textwrap.fill(encoded_text, width=80) # Adjust width as needed
                    pdf.multi_cell(0, 10, wrapped_text)
                except UnicodeEncodeError as enc_err:
                     print(f"PDF Encoding Error (Chunk {internal_chunk_number}): {enc_err}. Skipping problematic characters.")
                     # Fallback: try replacing errors
                     safe_text = pdf_text.encode('latin-1', 'ignore').decode('latin-1')
                     wrapped_text = textwrap.fill(safe_text, width=80)
                     pdf.multi_cell(0, 10, wrapped_text)
                except Exception as pdf_cell_err:
                     print(f"Error adding text to PDF (Chunk {internal_chunk_number}): {pdf_cell_err}")


                print(f"Saving PDF for chunk {internal_chunk_number}...")
                pdf_path = os.path.join(OUTPUT_DIR, pdf_filename)
                try:
                    pdf.output(pdf_path)
                    print(f"Conversation chunk saved: {pdf_path}")
                except Exception as pdf_save_err:
                    print(f"Error saving PDF {pdf_path}: {pdf_save_err}")
            else:
                print(f"No transcription detected for chunk {internal_chunk_number}, skipping PDF save.")

            print(f"=== Recognition Chunk {internal_chunk_number} finished. ===")
            internal_chunk_number += 1
            # Brief pause before starting next chunk? Optional.
            # time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nUser requested stop. Setting global stop flag...")
        stop_recognition_sep.set()
        stop_current_chunk.set() # Ensure current chunk processing also stops
    except Exception as e:
         print(f"An unexpected error occurred in the main loop: {e}")
         stop_recognition_sep.set() # Stop everything on unexpected error
         stop_current_chunk.set()
    finally:
        print("Cleaning up resources...")
        # Ensure stop flags are set
        if not stop_recognition_sep.is_set(): stop_recognition_sep.set()
        if not stop_current_chunk.is_set(): stop_current_chunk.set()

        # Stop PyAudio stream
        if stream is not None and stream.is_active():
            print("Stopping PyAudio stream...")
            stream.stop_stream()
        if stream is not None:
            print("Closing PyAudio stream...")
            stream.close()
        if audio is not None:
            print("Terminating PyAudio...")
            audio.terminate()
        print("PyAudio resources released.")

        # Final check/clearing of queues (might be redundant but safe)
        print("Final queue clearing...")
        queues_to_clear = [audio_queue, left_channel_queue, right_channel_queue]
        for q in queues_to_clear:
            with q.mutex: q.queue.clear()

        print("Main recognition loop finished or interrupted.")