import tempfile
from pathlib import Path
from openai import OpenAI
import pyaudio
from pydub import AudioSegment
from pydub.playback import play
import threading


_OUTPUT_DEVICE_AVAILABLE_CACHE: bool | None = None


def _has_default_output_device() -> bool:
    global _OUTPUT_DEVICE_AVAILABLE_CACHE

    if _OUTPUT_DEVICE_AVAILABLE_CACHE is not None:
        return _OUTPUT_DEVICE_AVAILABLE_CACHE

    pa = pyaudio.PyAudio()
    try:
        pa.get_default_output_device_info()
        _OUTPUT_DEVICE_AVAILABLE_CACHE = True
    except Exception:
        _OUTPUT_DEVICE_AVAILABLE_CACHE = False
    finally:
        pa.terminate()

    return _OUTPUT_DEVICE_AVAILABLE_CACHE


def speak_text(text):
    """Use OpenAI API to convert text to Speech."""
    cleaned_text = (text or "").strip()
    if not cleaned_text:
        return

    if not _has_default_output_device():
        print("OpenAI TTS skipped: no default output audio device found.")
        return

    try:
        client = OpenAI()

        tmp_dir = tempfile.gettempdir()
        print(f"Temporary directory: {tmp_dir}")
        speech_file_path = Path(tmp_dir) / "speech.mp3"

        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="coral",
            input=cleaned_text,
        ) as response:
            response.stream_to_file(speech_file_path)

        sound = AudioSegment.from_mp3(str(speech_file_path))
        play(sound)
    except Exception as exc:
        print(f"OpenAI TTS playback error: {exc}")
    
def speak_text_thread(text):
    # Play the text in a separate thread to avoid blocking
    thread = threading.Thread(target=speak_text, args=(text,), daemon=True)
    thread.start()

# Example usage:
# speak_text("Hello, this is a test of the OpenAI TTS system.")
# print()