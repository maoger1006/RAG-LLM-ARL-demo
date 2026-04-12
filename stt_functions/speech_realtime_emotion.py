"""AssemblyAI-backed compatibility layer for conversation STT imports.

Emotion tagging was removed from this module to eliminate legacy provider dependencies.
"""

from stt_functions.assemblyai_stt import (
    continuous_recognition,
    stop_recognition,
)

__all__ = [
    "continuous_recognition",
    "stop_recognition",
]
