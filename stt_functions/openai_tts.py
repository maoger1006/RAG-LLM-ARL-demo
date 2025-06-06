import tempfile
from pathlib import Path
from openai import OpenAI
from playsound import playsound
from pydub import AudioSegment
from pydub.playback import play
import threading


def speak_text(text):
    """Use OpenAI API to convert text to Speech."""
    
    # OpenAI client
    client = OpenAI()
    
    # Create a temporary directory for the speech file
    tmp_dir = tempfile.gettempdir()
    print(f"Temporary directory: {tmp_dir}")
    speech_file_path = Path(tmp_dir) / "speech.mp3"
    
    # Call OpenAI API
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="coral",
        input=text
    ) as response:
        response.stream_to_file(speech_file_path)
    
    # Use pydub to play the audio file
    sound = AudioSegment.from_mp3(str(speech_file_path))
    play(sound)
    
def speak_text_thread(text):
    # Play the text in a separate thread to avoid blocking
    thread = threading.Thread(target=speak_text, args=(text,))
    thread.start()

# Example usage:
# speak_text("Hello, this is a test of the OpenAI TTS system.")
# print()