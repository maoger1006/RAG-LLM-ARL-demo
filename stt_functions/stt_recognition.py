"""AssemblyAI-backed compatibility layer for legacy STT imports."""

from stt_functions.assemblyai_stt import (
    listen_and_recognize_multi,
    stop_listening_event,
    stt_for_query,
)

__all__ = [
    "listen_and_recognize_multi",
    "stop_listening_event",
    "stt_for_query",
]
