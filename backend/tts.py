"""Server-side TTS synthesis, browser-side playback.

The PyQt version played audio on the machine running the code; the web
version synthesizes an mp3 with OpenAI TTS, caches it in memory and
publishes a {type:"speak", url} event so every connected browser plays it
locally (important when the backend runs on a remote/SSH machine).
"""

import os
import threading
import uuid

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=".env", override=True)

_CACHE = {}     # tts_id -> mp3 bytes
_ORDER = []     # insertion order for eviction
_MAX_ITEMS = 50
_LOCK = threading.Lock()


def synthesize(text: str) -> str:
    """Generate speech for text, cache the mp3, return its cache id."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="coral",
        input=text,
    )
    audio_bytes = response.content

    tts_id = uuid.uuid4().hex
    with _LOCK:
        _CACHE[tts_id] = audio_bytes
        _ORDER.append(tts_id)
        while len(_ORDER) > _MAX_ITEMS:
            _CACHE.pop(_ORDER.pop(0), None)
    return tts_id


def get_audio(tts_id: str):
    with _LOCK:
        return _CACHE.get(tts_id)


def speak(state, bus, text):
    """Synthesize in a worker thread and tell browsers to play the result."""

    def run():
        try:
            tts_id = synthesize(text)
            bus.publish({
                "type": "speak",
                "url": f"/api/tts/{tts_id}",
                "text": str(text),
                "replay": False,
            })
        except Exception as e:
            print(f"TTS error: {e}")

    threading.Thread(target=run, daemon=True).start()
