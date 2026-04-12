"""AssemblyAI-backed compatibility layer for legacy realtime STT imports."""

from stt_functions.assemblyai_stt import (
    continuous_recognition,
    stop_recognition,
)


# Legacy symbol kept for compatibility with older imports.
def recog_stream(current_chunk_number, update_callback):
    return continuous_recognition(current_chunk_number, update_callback)


__all__ = [
    "continuous_recognition",
    "stop_recognition",
    "recog_stream",
]
