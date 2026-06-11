"""Browser-microphone audio gateway.

The browser captures the microphone (or shared system audio), downsamples
to 16 kHz PCM16 mono and streams the frames to the /ws/audio WebSocket.
Each connection runs one Google Cloud Speech streaming recognizer on the
server; recognized text is routed according to the session target:

  target "transcript"    -> Transcripts panel (Start STT button), channel 1
                            is labeled User, channel 2 (shared system audio)
                            is labeled Speaker and goes through the
                            content-correction logic
  target "question"      -> fills the question box (push-to-talk), emits a
                            final voice_done event when the stream ends
  target "conversation"  -> every final result is submitted as a question;
                            answers come back as qa + speak events

This replaces the PyAudio capture of the PyQt version, which only worked
on the machine running the code.
"""

import glob
import os
import queue
import threading
import time
from collections import deque

from backend import core, emotion, tts

SAMPLE_RATE = 16000
STREAM_LIMIT_SECONDS = 240  # restart the Google stream before the API limit


class GoogleRecognizer(threading.Thread):
    """Consume PCM16 frames from a queue and emit recognized text."""

    def __init__(self, on_final, on_interim=None, language="en-US"):
        super().__init__(daemon=True)
        from google.cloud import speech
        from google.oauth2 import service_account

        api_dir = os.path.join(os.getcwd(), "api")
        json_files = glob.glob(os.path.join(api_dir, "*.json"))
        if not json_files:
            raise FileNotFoundError("No JSON credential file was found in the ./api directory.")

        self.speech = speech
        credentials = service_account.Credentials.from_service_account_file(json_files[0])
        self.client = speech.SpeechClient(credentials=credentials)
        self.language = language
        self.on_final = on_final
        self.on_interim = on_interim

        self.audio_queue = queue.Queue()
        self._ended = threading.Event()

    def feed(self, data: bytes):
        if not self._ended.is_set():
            self.audio_queue.put(data)

    def end(self):
        """No more audio: flush pending results and stop."""
        self._ended.set()
        self.audio_queue.put(None)

    def _request_generator(self, deadline):
        speech = self.speech
        while time.time() < deadline:
            try:
                data = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                if self._ended.is_set():
                    return
                continue
            if data is None:
                return
            yield speech.StreamingRecognizeRequest(audio_content=data)

    def run(self):
        speech = self.speech
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE,
            language_code=self.language,
            enable_automatic_punctuation=True,
        )
        streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True)

        # outer loop restarts the stream below the ~5 min API limit
        while True:
            deadline = time.time() + STREAM_LIMIT_SECONDS
            try:
                responses = self.client.streaming_recognize(
                    streaming_config, self._request_generator(deadline)
                )
                for response in responses:
                    interim_parts = []
                    for result in response.results:
                        text = result.alternatives[0].transcript
                        if result.is_final:
                            self.on_final(text)
                        else:
                            interim_parts.append(text)
                    if interim_parts and self.on_interim:
                        self.on_interim(" ".join(interim_parts))
            except Exception as e:
                print(f"Audio recognition error: {e}")
            if self._ended.is_set():
                break


class AudioSession:
    """One /ws/audio connection: lifecycle + routing of recognized text."""

    def __init__(self, state, bus, target, channel=1):
        self.state = state
        self.bus = bus
        self.target = target
        self.channel = int(channel or 1)
        self._finals = []
        self._closed = False

        # rolling window of the most recent audio, analyzed by Hume.ai for
        # each final result in conversation mode (RAG-ARL-updates port)
        self._emotion_enabled = target == "conversation" and emotion.available()
        self._ring = deque()
        self._ring_bytes = 0

        if target == "transcript":
            if self.channel == 1:
                state.use_retrieval = True  # same auto-switch as the Qt Start STT
                state.stt_running = True
                core.publish_settings(state, bus)
                core.system_msg(bus, "transcripts", "Listening...")
            on_final = self._on_transcript_final
            on_interim = None
        elif target == "question":
            core.system_msg(bus, "transcripts", "Listening for voice input...")
            on_final = self._on_question_final
            on_interim = self._on_question_interim
        elif target == "conversation":
            state.conversation_running = True
            state.read_aloud = True
            core.publish_settings(state, bus)
            core.system_msg(bus, "qa", "Conversation started.")
            tts.speak(state, bus, "Hello, how can I help you today?")
            on_final = self._on_conversation_final
            on_interim = None
        else:
            raise ValueError(f"Unknown audio target: {target}")

        self.recognizer = GoogleRecognizer(on_final=on_final, on_interim=on_interim)
        self.recognizer.start()

    # ------------------------------------------------------------ routing

    def _on_transcript_final(self, text):
        core.handle_transcript_line(self.state, self.bus, text, self.channel)

    def _on_question_interim(self, text):
        combined = " ".join(self._finals + [text]).strip()
        self.bus.publish({"type": "voice_transcript", "text": combined, "replay": False})

    def _on_question_final(self, text):
        self._finals.append(text.strip())
        combined = " ".join(self._finals).strip()
        self.bus.publish({"type": "voice_transcript", "text": combined, "replay": False})

    def _on_conversation_final(self, text):
        user_text = text.strip()
        if not user_text:
            return
        if self._emotion_enabled and self._ring:
            label = emotion.detect(list(self._ring))
            if label:
                user_text = f"{user_text}  [Emotion: {label}]"
        self.bus.publish({"type": "voice_transcript", "text": user_text, "replay": False})
        try:
            core.answer_question(self.state, self.bus, user_text)
        except Exception as e:
            core.system_msg(self.bus, "qa", f"Error generating answer: {e}")

    # ------------------------------------------------------------ lifecycle

    def feed(self, data: bytes):
        if self._emotion_enabled:
            self._ring.append(data)
            self._ring_bytes += len(data)
            while self._ring_bytes > emotion.WINDOW_BYTES:
                self._ring_bytes -= len(self._ring.popleft())
        self.recognizer.feed(data)

    def close(self):
        """Blocking: flush the recognizer, then finalize per target."""
        if self._closed:
            return
        self._closed = True
        self.recognizer.end()
        self.recognizer.join(timeout=10)

        state, bus = self.state, self.bus
        if self.target == "transcript" and self.channel == 1:
            if state.history_transcript.strip():
                try:
                    core.save_transcription(state.history_transcript, state.current_chunk_number)
                    core.system_msg(
                        bus, "transcripts",
                        f"Transcription saved as transcription_chunk_{state.current_chunk_number}.pdf",
                    )
                    state.current_chunk_number += 1
                except Exception as e:
                    core.system_msg(bus, "transcripts", f"Error during transcription: {e}")
            state.history_transcript = ""
            state.stt_running = False
            core.publish_settings(state, bus)
            core.refresh_rag(state, bus)
        elif self.target == "question":
            core.system_msg(bus, "transcripts", "Voice input stopped. Processing...")
            bus.publish({
                "type": "voice_done",
                "text": " ".join(self._finals).strip(),
                "replay": False,
            })
        elif self.target == "conversation":
            state.conversation_running = False
            core.publish_settings(state, bus)
