"""
real_time_stt_with_emotion.py
─────────────────────────────
• Google Cloud Speech-to-Text streaming (continuous 240 s chunks)
• Rolling 5-second audio ring-buffer
• As-you-go PDF transcript saving
• Non-blocking Hume.ai prosody analysis on the most recent 5 s
─────────────────────────────
Before running:
  1.  pip install pyaudio fpdf google-cloud-speech hume
  2.  export GOOGLE_APPLICATION_CREDENTIALS=./api/your-gcloud-creds.json
  3.  export HUME_API_KEY=your_hume_api_key
"""

import os, sys, glob, queue, threading, textwrap, tempfile, wave, asyncio
from collections import deque
from datetime import datetime

import pyaudio
from fpdf import FPDF
from google.cloud import speech
from google.oauth2 import service_account

import base64, aiofiles, tempfile, os
from contextlib import asynccontextmanager

# ─────────────────────── Configuration ──────────────────────── #

RATE  = 16_000
CHUNK = RATE // 10                  # 100 ms (1 CHUNK == 100 ms of mono audio)
CHUNK_SECONDS      = 240            # length of one Google STT stream
LAST_SECONDS       = 5              # length of audio passed to Hume.ai
PDF_OUTDIR         = "./source"

# ─────────────────────── Credentials ────────────────────────── #

api_dir    = os.path.join(os.path.dirname(os.path.dirname(__file__)), "api")
json_files = glob.glob(os.path.join(api_dir, "*.json"))
if not json_files:
    raise FileNotFoundError("No JSON credential file was found in the ./api directory.")
credential = service_account.Credentials.from_service_account_file(json_files[0])
HUME_API_KEY = "" # Replace with your Hume API key or set as an environment variable
if not HUME_API_KEY:
    raise EnvironmentError("Set HUME_API_KEY in your environment.")

# ─────────────────────── Global state ───────────────────────── #

stop_recognition   = threading.Event()
chunk_stop         = threading.Event()
audio_queue        = queue.Queue()
ring_buffer        = deque(maxlen=int(LAST_SECONDS / (CHUNK / RATE)))  # 5 s window
transcription_lock = threading.Lock()
global_transcription = ""

# Async loop for non-blocking emotion calls
emotion_loop = asyncio.new_event_loop()
def _run_emotion_loop(loop): asyncio.set_event_loop(loop); loop.run_forever()
threading.Thread(target=_run_emotion_loop, args=(emotion_loop,), daemon=True).start()

# ──────────────────── Hume.ai helper ────────────────────────── #

from hume import AsyncHumeClient
from hume.expression_measurement.stream  import Config
from hume.expression_measurement.stream.socket_client import StreamConnectOptions
hume_client = AsyncHumeClient(api_key=HUME_API_KEY)
hume_model_config = Config(prosody={})
hume_stream_opts  = StreamConnectOptions(config=hume_model_config)

async def analyse_emotion_async(audio_bytes: bytes) -> str:
    """
    Push 5-second audio to Hume.ai and return the top-scoring emotion
    (or "not detected" on failure / silence).
    """

    try:
        async with hume_client.expression_measurement.stream.connect(
            options=hume_stream_opts
        ) as socket:

            # ── 1. choose the right send-method ─────────────────────────
            # if hasattr(socket, "send_bytes"):               # old StreamSocket
            #     b64 = base64.b64encode(audio_bytes)
            #     result = await socket.send_bytes(b64)
            # else:                                           # new Connection
            # write a temp WAV and use send_file
            async with aiofiles.tempfile.NamedTemporaryFile(
                delete=False, suffix=".wav"
            ) as fp:
                await fp.write(audio_bytes)
                tmp_path = fp.name
            try:
                result = await socket.send_file(tmp_path)
            finally:
                os.remove(tmp_path)

        # ── 2. pull out prosody predictions regardless of SDK shape ─────
        preds = None
        if hasattr(result, "prosody"):                      # object style
            preds = result.prosody.predictions
        elif isinstance(result, dict) and "prosody" in result:  # dict style
            preds = result["prosody"]["predictions"]

        if preds:
            emotions = preds[0]["emotions"] if isinstance(preds[0], dict) else preds[0].emotions
            if emotions:
                top = max(
                    emotions,
                    key=lambda e: e["score"] if isinstance(e, dict) else e.score
                )
                return top["name"] if isinstance(top, dict) else top.name

    except Exception as exc:
        print("Hume.ai error:", exc)

    return "Neutral"  # default if no emotion detected

def submit_emotion_task(audio_bytes: bytes):
    """Run the coroutine on the background event-loop and return its Future."""
    return asyncio.run_coroutine_threadsafe(
        analyse_emotion_async(audio_bytes), emotion_loop
    )

# ──────────────────── Audio helpers ─────────────────────────── #

def build_wav_bytes(frames: list[bytes]) -> bytes:
    """Return a little-endian 16-bit mono WAV in-memory bytes from raw frames."""
    with tempfile.TemporaryFile() as fp:
        with wave.open(fp, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)          # 16-bit
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
        fp.seek(0)
        return fp.read()

# ──────────────────── PyAudio callback ──────────────────────── #

def audio_callback(in_data, frame_count, time_info, status):
    if stop_recognition.is_set():
        return None, pyaudio.paComplete
    audio_queue.put(in_data)
    ring_buffer.append(in_data)          # keep rolling 5 s window
    return None, pyaudio.paContinue

# ──────────────────── Main STT loop ─────────────────────────── #

def continuous_recognition(chunk_idx, update_callback=lambda _: None):
    """Loop forever, restarting Google STT every CHUNK_SECONDS."""
    global global_transcription
    internal_idx = 0
    
    with transcription_lock:
        global_transcription = ""

    try:
        while not stop_recognition.is_set():
            print(f"\n=== start chunk {internal_idx} (every {CHUNK_SECONDS}s) ===")

            recog_thread = threading.Thread(
                target=recognise_stream,
                args=(chunk_idx, update_callback),
                daemon=True
            )
            timer = threading.Timer(CHUNK_SECONDS, lambda: chunk_stop.set())

            chunk_stop.clear()
            recog_thread.start()
            timer.start()

            recog_thread.join()
            timer.cancel()

            with audio_queue.mutex:
                audio_queue.queue.clear()
                
            internal_idx += 1

    except KeyboardInterrupt:
        print("User requested stop.")
        stop_recognition.set()        
        recog_thread.join(timeout=1)
        timer.cancel()

def recognise_stream(chunk_number: int, update_callback):
    """One 240-s Google STT session."""
    global global_transcription
    global chunk_stop
    # ----- PDF setup -----
    os.makedirs(PDF_OUTDIR, exist_ok=True)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Conversation - chunk {chunk_number}", ln=True, align="C")

    # ----- Google STT client -----
    client = speech.SpeechClient(credentials=credential)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US",
        enable_automatic_punctuation=True,
    )
    streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True)

    # ----- PyAudio -----
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK,
                        stream_callback=audio_callback)

    def gen_audio():
        while not chunk_stop.is_set() and not stop_recognition.is_set():
            try:
                data = audio_queue.get(timeout=1)
                yield speech.StreamingRecognizeRequest(audio_content=data)
            except queue.Empty:
                continue

    print("🎙️  Listening…")
    responses = client.streaming_recognize(streaming_config, gen_audio())

    try:
        for response in responses:
            for result in response.results:

                if not result.is_final:
                    continue

                raw_text = result.alternatives[0].transcript.strip()

                # ────── NEW: ignore silence / empty transcripts ──────
                if not raw_text:
                    continue

                # ---- snapshot last 5-s audio & launch emotion call --
                emotion_fut = None
                if ring_buffer:
                    wav_bytes  = build_wav_bytes(list(ring_buffer))
                    emotion_fut = submit_emotion_task(wav_bytes)   # async

                # ---- callback to merge emotion once Hume replies ----
                def _attach_emotion(fut, text=raw_text):
                    global global_transcription
                    try:
                        label = fut.result() if fut else "neutral"
                    except Exception:
                        label = "neutral"

                    tagged = f"{text}  [Emotion: {label}]"

                    with transcription_lock:
                        global_transcription += tagged + '\n'

                    update_callback(tagged)

                if emotion_fut:
                    emotion_fut.add_done_callback(_attach_emotion)
                else:                         # ring_buffer was empty (very start)
                    _attach_emotion(None)

            # graceful exit on chunk_stop
            if chunk_stop.is_set():
                break
    except Exception as e:
        print("Google STT error:", e)

    finally:
        stream.stop_stream(); stream.close(); audio.terminate()

        # ---- Launch emotion analysis on the last 5 s ----
        # emotion_label = "neutral"
        # if ring_buffer:
        #     wav_bytes = build_wav_bytes(list(ring_buffer))
        #     fut = submit_emotion_task(wav_bytes)      # enqueue on event-loop
        #     try:
        #         # Wait up to 15 s for Hume.ai to reply
        #         emotion_label = fut.result(timeout=15)
        #     except Exception as exc:
        #         print("⚠️  Emotion task failed:", exc)

        # ---- Combine transcript + emotion and hand it out ----
        if global_transcription.strip():
            # tagged = f"{global_transcription.strip()}  [Emotion: {emotion_label}]"
            tagged = global_transcription.strip()
            wrapped = textwrap.fill(tagged, width=90)
            pdf.multi_cell(0, 10, wrapped)

            fname = f"conversation_chunk_{chunk_number}.pdf"
            pdf.output(os.path.join(PDF_OUTDIR, fname))
            print(f"📝 Transcript saved → {fname}")

            # send to any UI callback
            # update_callback(tagged)

        print("⏹️  Recognition stopped for this chunk.")

# ───────────────────────── Runner ───────────────────────────── #

if __name__ == "__main__":
    print("Ctrl-C to quit.\n")
    continuous_recognition()
