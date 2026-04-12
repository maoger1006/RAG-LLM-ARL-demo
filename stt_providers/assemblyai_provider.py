import json
import os
import tempfile
import threading
import time
from typing import Callable, Dict, Optional
from urllib.parse import urlencode

from dotenv import load_dotenv
import requests
import websocket


load_dotenv()

API_BASE_URL = os.getenv("ASSEMBLY_AI_BASE_URL", "https://api.assemblyai.com")
STREAMING_BASE_URL = os.getenv("ASSEMBLY_AI_STREAM_URL", "wss://streaming.assemblyai.com/v3/ws")
DEFAULT_SPEECH_MODELS = ["universal-3-pro", "universal-2"]
DEFAULT_STREAMING_MODEL = "u3-rt-pro"


class AssemblyAIError(RuntimeError):
    pass


class AssemblyAIClient:
    """AssemblyAI pre-recorded transcription helper using the REST API."""

    def __init__(self, api_key: Optional[str] = None, base_url: str = API_BASE_URL):
        self.api_key = (api_key or os.getenv("ASSEMBLY_AI_KEY") or "").strip()
        if not self.api_key:
            raise EnvironmentError("Missing ASSEMBLY_AI_KEY environment variable.")

        self.base_url = base_url.rstrip("/")
        self.headers = {"authorization": self.api_key}

    def upload_file(self, file_path: str) -> str:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        with open(file_path, "rb") as audio_f:
            response = requests.post(
                f"{self.base_url}/v2/upload",
                headers=self.headers,
                data=audio_f,
                timeout=120,
            )

        if response.status_code != 200:
            raise AssemblyAIError(
                f"AssemblyAI upload failed ({response.status_code}): {response.text}"
            )

        return response.json()["upload_url"]

    def transcribe_file(
        self,
        file_path: str,
        speaker_labels: bool = False,
        speakers_expected: Optional[int] = None,
        language_detection: bool = True,
        poll_interval_seconds: float = 1.5,
    ) -> Dict:
        audio_url = self.upload_file(file_path)

        payload = {
            "audio_url": audio_url,
            "speech_models": DEFAULT_SPEECH_MODELS,
            "language_detection": language_detection,
            "speaker_labels": speaker_labels,
        }

        if speakers_expected and speakers_expected > 0:
            payload["speakers_expected"] = int(speakers_expected)

        response = requests.post(
            f"{self.base_url}/v2/transcript",
            headers=self.headers,
            json=payload,
            timeout=60,
        )

        # Some accounts/endpoints may reject optional diarization fields; retry without it.
        if response.status_code != 200 and "speakers_expected" in payload:
            payload.pop("speakers_expected", None)
            response = requests.post(
                f"{self.base_url}/v2/transcript",
                headers=self.headers,
                json=payload,
                timeout=60,
            )

        if response.status_code != 200:
            raise AssemblyAIError(
                f"AssemblyAI submit failed ({response.status_code}): {response.text}"
            )

        transcript_id = response.json()["id"]
        polling_endpoint = f"{self.base_url}/v2/transcript/{transcript_id}"

        while True:
            transcript = requests.get(
                polling_endpoint,
                headers=self.headers,
                timeout=60,
            ).json()

            status = transcript.get("status")
            if status == "completed":
                return transcript

            if status == "error":
                raise AssemblyAIError(
                    f"AssemblyAI transcription failed: {transcript.get('error', 'unknown error')}"
                )

            time.sleep(poll_interval_seconds)

    def transcribe_pcm_bytes(
        self,
        pcm_audio_bytes: bytes,
        sample_rate: int,
        channels: int = 1,
        sample_width: int = 2,
        speaker_labels: bool = False,
        speakers_expected: Optional[int] = None,
    ) -> Dict:
        import wave

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            with wave.open(temp_path, "wb") as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(sample_rate)
                wf.writeframes(pcm_audio_bytes)

            return self.transcribe_file(
                temp_path,
                speaker_labels=speaker_labels,
                speakers_expected=speakers_expected,
            )
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass


class RealtimeAssemblyAISession:
    """Streaming transcription session via AssemblyAI's realtime WebSocket API."""

    def __init__(
        self,
        sample_rate: int,
        on_final: Callable[[str], None],
        on_partial: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        api_key: Optional[str] = None,
        speech_model: str = DEFAULT_STREAMING_MODEL,
    ):
        self.api_key = (api_key or os.getenv("ASSEMBLY_AI_KEY") or "").strip()
        if not self.api_key:
            raise EnvironmentError("Missing ASSEMBLY_AI_KEY environment variable.")

        self.sample_rate = sample_rate
        self.speech_model = speech_model
        self.on_final = on_final
        self.on_partial = on_partial
        self.on_error = on_error

        self._ws_app: Optional[websocket.WebSocketApp] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._connected = threading.Event()
        self._closed = threading.Event()
        self._send_lock = threading.Lock()

    def _build_url(self) -> str:
        params = {
            "sample_rate": self.sample_rate,
            "speech_model": self.speech_model,
        }
        return f"{STREAMING_BASE_URL}?{urlencode(params)}"

    def _handle_message(self, _ws, message: str) -> None:
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return

        msg_type = data.get("type")

        if msg_type == "Turn":
            transcript = (data.get("transcript") or "").strip()
            if not transcript:
                return

            if data.get("end_of_turn"):
                self.on_final(transcript)
            elif self.on_partial:
                self.on_partial(transcript)
            return

        if msg_type in {"Error", "Termination"} and self.on_error:
            detail = data.get("message") or data.get("error") or str(data)
            self.on_error(AssemblyAIError(detail))

    def _handle_error(self, _ws, error) -> None:
        if self.on_error:
            if isinstance(error, Exception):
                self.on_error(error)
            else:
                self.on_error(AssemblyAIError(str(error)))

    def _handle_open(self, _ws) -> None:
        self._connected.set()

    def _handle_close(self, _ws, _status_code, _message) -> None:
        self._closed.set()

    def start(self, timeout_seconds: float = 10.0) -> None:
        self._connected.clear()
        self._closed.clear()

        self._ws_app = websocket.WebSocketApp(
            self._build_url(),
            header=[f"Authorization: {self.api_key}"],
            on_open=self._handle_open,
            on_message=self._handle_message,
            on_error=self._handle_error,
            on_close=self._handle_close,
        )

        self._ws_thread = threading.Thread(
            target=self._ws_app.run_forever,
            kwargs={"ping_interval": 20, "ping_timeout": 10},
            daemon=True,
        )
        self._ws_thread.start()

        if not self._connected.wait(timeout=timeout_seconds):
            self.close(send_terminate=False)
            raise AssemblyAIError("Failed to establish AssemblyAI realtime session.")

    def stream(self, audio_chunk: bytes) -> None:
        if not audio_chunk:
            return

        if not self._ws_app or not self._ws_app.sock or not self._ws_app.sock.connected:
            raise AssemblyAIError("Realtime session is not connected.")

        with self._send_lock:
            self._ws_app.send(audio_chunk, opcode=websocket.ABNF.OPCODE_BINARY)

    def close(self, send_terminate: bool = True, timeout_seconds: float = 3.0) -> None:
        if not self._ws_app:
            return

        if send_terminate and self._ws_app.sock and self._ws_app.sock.connected:
            try:
                with self._send_lock:
                    self._ws_app.send(json.dumps({"type": "Terminate"}))
            except Exception:
                pass

        try:
            self._ws_app.close()
        except Exception:
            pass

        self._closed.wait(timeout=timeout_seconds)

        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=timeout_seconds)
