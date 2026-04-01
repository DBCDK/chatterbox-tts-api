"""
Core functionality for Chatterbox TTS API
"""

from .memory import get_memory_info, cleanup_memory, safe_delete_tensors
from .text_processing import (
    split_text_into_chunks,
    concatenate_audio_chunks,
    split_text_for_streaming,
    get_streaming_settings,
)
from .tts_model import initialize_model, get_model
from .version import get_version, get_version_info

__all__ = [
    "get_memory_info",
    "cleanup_memory",
    "safe_delete_tensors",
    "split_text_into_chunks",
    "concatenate_audio_chunks",
    "split_text_for_streaming",
    "get_streaming_settings",
    "initialize_model",
    "get_model",
    "get_version",
    "get_version_info",
]
