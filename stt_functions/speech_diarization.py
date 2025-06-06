from google.cloud import speech
from google.oauth2 import service_account
import os
import glob

# Load Google Cloud credentials
# Look for a JSON file in the "./api" directory and use the first one found
api_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "api")
json_files = glob.glob(os.path.join(api_dir, "*.json"))

if not json_files:
    raise FileNotFoundError("No JSON credential file was found in the ./api directory.")

credentials = service_account.Credentials.from_service_account_file(json_files[0])
client = speech.SpeechClient(credentials=credentials)

# Load audio file
audio_file_path = "./test audio/meeting.wav"
with open(audio_file_path, "rb") as audio_file:
    audio = speech.RecognitionAudio(content=audio_file.read())

# Set recognition config with speaker diarization
config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=48000,  # Ensure this matches your audio file's sample rate
    language_code="en-US",
    enable_automatic_punctuation=True,  # Optional: Improves readability
    diarization_config=speech.SpeakerDiarizationConfig(
        enable_speaker_diarization=True,
        min_speaker_count=2,
        max_speaker_count=2  # Adjust based on expected number of speakers
    ),
    model="latest_long"  # Use a more accurate model for long-form audio
)

# Perform speech-to-text with diarization
response = client.recognize(config=config, audio=audio)

# Perform speech-to-text with diarization
response = client.recognize(config=config, audio=audio)

# Group words by speaker
transcript_by_speaker = []
current_speaker = None
current_text = []

for result in response.results:
    words_info = result.alternatives[0].words  # Get words with speaker info
    for word in words_info:
        speaker = word.speaker_tag

        # If speaker changes, store the last spoken sentence
        if speaker != current_speaker:
            if current_text:
                transcript_by_speaker.append(f"Speaker {current_speaker}: {' '.join(current_text)}")
                current_text = []

            current_speaker = speaker  # Update speaker

        # Append word to the current speaker's sentence
        current_text.append(word.word)

# Append the last speaker's words
if current_text:
    transcript_by_speaker.append(f"Speaker {current_speaker}: {' '.join(current_text)}")

# Print formatted transcript
for line in transcript_by_speaker:
    print(line)
