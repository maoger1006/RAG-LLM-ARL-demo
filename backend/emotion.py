"""Hume.ai speech-prosody emotion detection for the conversation mode.

Port of stt_functions/speech_realtime_emotion.py from RAG-ARL-updates:
the last few seconds of microphone audio are sent to the Hume.ai
expression-measurement API and the top-scoring emotion label is appended
to the recognized question as "[Emotion: <label>]" so the LLM can adapt
its answer to how the user sounds.

Requires HUME_API_KEY in the environment (.env). Without the key (or
without the `hume` package installed) detection is disabled and the
conversation behaves exactly as before.
"""

import asyncio
import io
import os
import tempfile
import threading
import wave

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)

SAMPLE_RATE = 16000
WINDOW_SECONDS = 5  # rolling audio context sent to Hume per request
WINDOW_BYTES = WINDOW_SECONDS * SAMPLE_RATE * 2  # PCM16 mono

_lock = threading.Lock()
_loop = None
_client = None
_disabled = False


def _get_client():
    """Lazy Hume client; returns None (and stays disabled) when unusable."""
    global _client, _disabled
    with _lock:
        if _client is not None or _disabled:
            return _client
        api_key = os.getenv("HUME_API_KEY", "").strip()
        if not api_key:
            _disabled = True
            print("Emotion detection disabled: HUME_API_KEY is not set.")
            return None
        try:
            from hume import AsyncHumeClient
        except ImportError:
            _disabled = True
            print("Emotion detection disabled: the `hume` package is not installed.")
            return None
        _client = AsyncHumeClient(api_key=api_key)
        return _client


def available() -> bool:
    return _get_client() is not None


def _get_loop():
    """Background event loop so Hume calls never touch the server loop."""
    global _loop
    with _lock:
        if _loop is None:
            loop = asyncio.new_event_loop()

            def runner():
                asyncio.set_event_loop(loop)
                loop.run_forever()

            threading.Thread(target=runner, daemon=True).start()
            _loop = loop
        return _loop


def build_wav_bytes(frames) -> bytes:
    """In-memory 16 kHz mono PCM16 WAV from raw audio frames."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))
    return buf.getvalue()


async def _analyse(wav_bytes: bytes):
    from hume.expression_measurement.stream import Config

    async with _get_client().expression_measurement.stream.connect() as socket:
        # the streaming SDK only accepts files, so round-trip through a temp WAV
        fd, path = tempfile.mkstemp(suffix=".wav")
        try:
            with os.fdopen(fd, "wb") as fp:
                fp.write(wav_bytes)
            result = await socket.send_file(path, config=Config(prosody={}))
        finally:
            os.remove(path)

    # prosody predictions arrive as objects or dicts depending on SDK version
    preds = None
    if getattr(result, "prosody", None) is not None:
        preds = result.prosody.predictions
    elif isinstance(result, dict) and "prosody" in result:
        preds = result["prosody"]["predictions"]
    if not preds:
        return None

    first = preds[0]
    emotions = first["emotions"] if isinstance(first, dict) else first.emotions
    if not emotions:
        return None
    top = max(emotions, key=lambda e: e["score"] if isinstance(e, dict) else e.score)
    return top["name"] if isinstance(top, dict) else top.name


def detect(frames, timeout=10):
    """Top emotion label for the given PCM16 frames, or None on any failure."""
    if not frames or not available():
        return None
    try:
        fut = asyncio.run_coroutine_threadsafe(_analyse(build_wav_bytes(frames)), _get_loop())
        return fut.result(timeout=timeout)
    except Exception as exc:
        print(f"Hume.ai emotion error: {exc}")
        return None
