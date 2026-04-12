import os
import threading
import time
from typing import Callable, Dict, List

from dotenv import load_dotenv
from fpdf import FPDF
import pyaudio

from stt_providers.assemblyai_provider import AssemblyAIClient, RealtimeAssemblyAISession


load_dotenv()

RATE = 16000
FORMAT = pyaudio.paInt16
CHUNK = int(RATE / 10)  # 100 ms
SOURCE_DIR = "./source"
CHUNK_SECONDS_SEPARATED = int(os.getenv("ASSEMBLY_AI_SEPARATED_CHUNK_SECONDS", "12"))
_INPUT_DEVICE_AVAILABLE_CACHE: bool | None = None

# These events are imported by gui_beta.py.
stop_listening_event = threading.Event()
stop_recognition_sep = threading.Event()
stop_recognition = threading.Event()


def _ensure_source_dir() -> None:
    os.makedirs(SOURCE_DIR, exist_ok=True)


def _get_client() -> AssemblyAIClient:
    return AssemblyAIClient()


def _has_default_input_device() -> bool:
    global _INPUT_DEVICE_AVAILABLE_CACHE

    if _INPUT_DEVICE_AVAILABLE_CACHE is not None:
        return _INPUT_DEVICE_AVAILABLE_CACHE

    pa = pyaudio.PyAudio()
    try:
        pa.get_default_input_device_info()
        _INPUT_DEVICE_AVAILABLE_CACHE = True
    except Exception:
        _INPUT_DEVICE_AVAILABLE_CACHE = False
    finally:
        pa.terminate()

    return _INPUT_DEVICE_AVAILABLE_CACHE


def _record_pcm(
    stop_event: threading.Event,
    duration_seconds: float | None = None,
    channels: int = 1,
) -> bytes:
    """Record PCM16 audio from the default input device."""
    if not _has_default_input_device():
        raise RuntimeError(
            "No default input audio device found. Configure a microphone before starting STT."
        )

    pa = pyaudio.PyAudio()
    stream = None
    frames: List[bytes] = []
    start_time = time.time()

    try:
        try:
            stream = pa.open(
                format=FORMAT,
                channels=channels,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
            )
        except OSError as exc:
            raise RuntimeError(
                "Unable to open the default input audio device. Configure a microphone and try again."
            ) from exc

        while not stop_event.is_set():
            if duration_seconds is not None and (time.time() - start_time) >= duration_seconds:
                break

            data = stream.read(CHUNK, exception_on_overflow=False)
            if data:
                frames.append(data)

    finally:
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        pa.terminate()

    return b"".join(frames)


def _transcribe_pcm(
    pcm_audio_bytes: bytes,
    speaker_labels: bool = False,
    speakers_expected: int | None = None,
) -> Dict:
    client = _get_client()
    return client.transcribe_pcm_bytes(
        pcm_audio_bytes=pcm_audio_bytes,
        sample_rate=RATE,
        channels=1,
        sample_width=2,
        speaker_labels=speaker_labels,
        speakers_expected=speakers_expected,
    )


def _speaker_channel(raw_speaker: str, speaker_map: Dict[str, int]) -> int:
    if raw_speaker not in speaker_map:
        speaker_map[raw_speaker] = 1 if len(speaker_map) == 0 else 2
    return speaker_map[raw_speaker]


def stt_for_query() -> str:
    """Record until stop_listening_event is set, then transcribe with AssemblyAI."""
    try:
        pcm_data = _record_pcm(stop_event=stop_listening_event, channels=1)
        if not pcm_data:
            return ""

        transcript = _transcribe_pcm(pcm_data, speaker_labels=False)
        return (transcript.get("text") or "").strip()
    except Exception as e:
        print(f"AssemblyAI query transcription error: {e}")
        return ""


def listen_and_recognize_multi(current_chunk_number, min_speaker_count=1, max_speaker_count=3):
    """Compatibility wrapper used by legacy call sites."""
    _ = min_speaker_count
    try:
        _ensure_source_dir()
        pcm_data = _record_pcm(stop_event=stop_listening_event, channels=1)
        if not pcm_data:
            return ""

        transcript = _transcribe_pcm(
            pcm_data,
            speaker_labels=True,
            speakers_expected=max_speaker_count,
        )

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        lines: List[str] = []
        speaker_map: Dict[str, int] = {}
        utterances = transcript.get("utterances") or []

        for utterance in utterances:
            text = (utterance.get("text") or "").strip()
            if not text:
                continue
            channel = _speaker_channel(str(utterance.get("speaker", "spk")), speaker_map)
            line = f"Speaker {channel}: {text}"
            lines.append(line)
            pdf.multi_cell(0, 8, line)

        if not lines:
            text = (transcript.get("text") or "").strip()
            if text:
                lines.append(text)
                pdf.multi_cell(0, 8, text)

        output_path = os.path.join(SOURCE_DIR, f"transcription_chunk_{current_chunk_number}.pdf")
        if lines:
            pdf.output(output_path)

        return "\n".join(lines)
    except Exception as e:
        print(f"AssemblyAI multi-speaker transcription error: {e}")
        return ""


def continuous_recognition_sep(initial_chunk_number=1, update_callback=lambda text, channel: None):
    """
    Chunked AssemblyAI transcription loop for dual-speaker style UI updates.
    Emits callbacks with (text, channel_tag) where channel_tag is 1 or 2.
    """
    current_chunk_number = int(initial_chunk_number)
    _ensure_source_dir()

    while not stop_recognition_sep.is_set():
        pcm_data = _record_pcm(
            stop_event=stop_recognition_sep,
            duration_seconds=CHUNK_SECONDS_SEPARATED,
            channels=1,
        )

        if not pcm_data:
            continue

        try:
            transcript = _transcribe_pcm(
                pcm_data,
                speaker_labels=True,
                speakers_expected=2,
            )
        except Exception as e:
            print(f"AssemblyAI separated transcription error: {e}")
            continue

        lines: List[str] = []
        utterances = transcript.get("utterances") or []
        speaker_map: Dict[str, int] = {}

        if utterances:
            for utterance in utterances:
                text = (utterance.get("text") or "").strip()
                if not text:
                    continue
                channel = _speaker_channel(str(utterance.get("speaker", "spk")), speaker_map)
                update_callback(text, channel)
                lines.append(f"Ch{channel}: {text}")
        else:
            text = (transcript.get("text") or "").strip()
            if text:
                update_callback(text, 1)
                lines.append(f"Ch1: {text}")

        if lines:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 8, "\n".join(lines))
            pdf.output(os.path.join(SOURCE_DIR, f"transcription_chunk_{current_chunk_number}.pdf"))

        current_chunk_number += 1


def continuous_recognition(chunk_idx, update_callback: Callable[[str], None] = lambda _: None):
    """Realtime AssemblyAI transcription loop for conversation mode."""
    _ = chunk_idx

    if not _has_default_input_device():
        print("AssemblyAI conversation loop error: No default input audio device found.")
        return

    session = None
    pa = pyaudio.PyAudio()
    stream = None

    def _on_final(text: str) -> None:
        cleaned = (text or "").strip()
        if cleaned:
            update_callback(cleaned)

    def _on_error(error: Exception) -> None:
        print(f"AssemblyAI realtime error: {error}")

    try:
        session = RealtimeAssemblyAISession(
            sample_rate=RATE,
            on_final=_on_final,
            on_error=_on_error,
        )
        session.start()

        stream = pa.open(
            format=FORMAT,
            channels=1,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

        while not stop_recognition.is_set():
            data = stream.read(CHUNK, exception_on_overflow=False)
            session.stream(data)

    except Exception as e:
        print(f"AssemblyAI conversation loop error: {e}")
    finally:
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        pa.terminate()

        if session is not None:
            try:
                session.close(send_terminate=True)
            except Exception:
                pass
