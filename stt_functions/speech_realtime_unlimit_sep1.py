"""AssemblyAI-backed compatibility layer for separated-channel STT imports."""

from stt_functions.assemblyai_stt import (
    continuous_recognition_sep,
    stop_recognition_sep,
)

__all__ = [
    "continuous_recognition_sep",
    "stop_recognition_sep",
]
