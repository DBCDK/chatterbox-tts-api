"""
TTS model initialization and management
"""

import asyncio
import os
import traceback
from enum import Enum
from typing import Any, Dict, Optional

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.tts import ChatterboxTTS
from huggingface_hub import snapshot_download

from app.config import Config, detect_device
from app.core.chatterbox_patches import apply_chatterbox_patches
from app.core.mtl import SUPPORTED_LANGUAGES

# Global model instance
_model = None
_device = None
_initialization_state = "not_started"
_initialization_error = None
_initialization_progress = ""
_is_multilingual = None
_supported_languages = {}
_model_metadata: Dict[str, Any] = {
    "model_source": "default",
    "model_class": None,
    "model_type": None,
    "model_repo_id": None,
    "model_revision": None,
    "model_local_path": None,
    "resolved_model_path": None,
    "default_language": "en",
}


class InitializationState(Enum):
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"


def _get_model_loader(model_class: str):
    if model_class == "multilingual":
        return ChatterboxMultilingualTTS
    if model_class == "standard":
        return ChatterboxTTS
    raise ValueError(f"Unsupported MODEL_CLASS: {model_class}")


def _resolve_supported_languages(model_source: str, model_class: str) -> Dict[str, str]:
    configured_languages = Config.get_configured_supported_languages()
    if configured_languages:
        return configured_languages.copy()
    if model_class == "multilingual" and model_source == "default":
        return SUPPORTED_LANGUAGES.copy()
    return {"en": "English"}


def _load_model_sync(
    model_source: str, model_class: str, device: str
) -> tuple[Any, Dict[str, Any]]:
    loader = _get_model_loader(model_class)
    metadata: Dict[str, Any] = {
        "model_source": model_source,
        "model_class": model_class,
        "model_type": model_class,
        "model_repo_id": Config.MODEL_REPO_ID or None,
        "model_revision": Config.MODEL_REVISION,
        "model_local_path": Config.MODEL_LOCAL_PATH,
        "resolved_model_path": None,
        "default_language": Config.get_default_language(),
    }

    if model_source == "default":
        model = loader.from_pretrained(device=device)
        return model, metadata

    if model_source == "hf_repo":
        resolved_model_path = snapshot_download(
            repo_id=Config.MODEL_REPO_ID,
            revision=Config.MODEL_REVISION,
            cache_dir=Config.MODEL_CACHE_DIR,
            token=Config.HF_TOKEN,
            allow_patterns=Config.get_hf_allow_patterns(),
        )
        metadata["resolved_model_path"] = resolved_model_path
        model = loader.from_local(resolved_model_path, device=device)
        return model, metadata

    if model_source == "local_dir":
        resolved_model_path = os.path.abspath(Config.MODEL_LOCAL_PATH)
        metadata["model_local_path"] = resolved_model_path
        metadata["resolved_model_path"] = resolved_model_path
        model = loader.from_local(resolved_model_path, device=device)
        return model, metadata

    raise ValueError(f"Unsupported MODEL_SOURCE: {model_source}")


async def initialize_model():
    """Initialize the Chatterbox TTS model"""
    global \
        _model, \
        _device, \
        _initialization_state, \
        _initialization_error, \
        _initialization_progress
    global _is_multilingual, _supported_languages, _model_metadata

    try:
        _initialization_state = InitializationState.INITIALIZING.value
        _initialization_progress = "Validating configuration..."

        Config.validate()
        apply_chatterbox_patches()
        _device = detect_device()
        model_source = Config.get_model_source()
        model_class = Config.get_model_class()
        default_language = Config.get_default_language()

        print("Initializing Chatterbox TTS model...")
        print(f"Device: {_device}")
        print(f"Voice sample: {Config.VOICE_SAMPLE_PATH}")
        print(f"Model cache: {Config.MODEL_CACHE_DIR}")
        print(f"Model source: {model_source}")
        print(f"Model class: {model_class}")
        if Config.MODEL_REPO_ID:
            print(f"Model repo: {Config.MODEL_REPO_ID}")
        if Config.MODEL_LOCAL_PATH:
            print(f"Model local path: {Config.MODEL_LOCAL_PATH}")

        _initialization_progress = "Creating model cache directory..."
        os.makedirs(Config.MODEL_CACHE_DIR, exist_ok=True)

        _initialization_progress = "Checking voice sample..."
        if not os.path.exists(Config.VOICE_SAMPLE_PATH):
            raise FileNotFoundError(
                f"Voice sample not found: {Config.VOICE_SAMPLE_PATH}"
            )

        if model_source == "local_dir" and not os.path.exists(Config.MODEL_LOCAL_PATH):
            raise FileNotFoundError(
                f"Model local path not found: {Config.MODEL_LOCAL_PATH}"
            )

        _initialization_progress = "Configuring device compatibility..."
        if _device == "cpu":
            import torch

            original_load = torch.load
            original_load_file = None

            try:
                import safetensors.torch

                original_load_file = safetensors.torch.load_file
            except ImportError:
                pass

            def force_cpu_torch_load(f, map_location=None, **kwargs):
                return original_load(f, map_location="cpu", **kwargs)

            def force_cpu_load_file(filename, device=None):
                return original_load_file(filename, device="cpu")

            torch.load = force_cpu_torch_load
            if original_load_file:
                safetensors.torch.load_file = force_cpu_load_file

        _initialization_progress = "Loading TTS model (this may take a while)..."
        loop = asyncio.get_event_loop()
        _model, model_metadata = await loop.run_in_executor(
            None, lambda: _load_model_sync(model_source, model_class, _device)
        )

        _is_multilingual = model_class == "multilingual"
        _supported_languages = _resolve_supported_languages(model_source, model_class)
        _model_metadata = {
            **model_metadata,
            "default_language": default_language,
        }

        print(f"Supported languages: {', '.join(_supported_languages.keys())}")
        if _model_metadata.get("resolved_model_path"):
            print(f"Resolved model path: {_model_metadata['resolved_model_path']}")

        _initialization_state = InitializationState.READY.value
        _initialization_progress = "Model ready"
        _initialization_error = None
        print(f"Model initialized successfully on {_device}")
        return _model

    except Exception as e:
        _initialization_state = InitializationState.ERROR.value
        _initialization_error = str(e)
        _initialization_progress = f"Failed: {str(e)}"
        print(f"Failed to initialize model: {e}")
        traceback.print_exc()
        raise e


def get_model():
    """Get the current model instance"""
    return _model


def get_device():
    """Get the current device"""
    return _device


def get_initialization_state():
    """Get the current initialization state"""
    return _initialization_state


def get_initialization_progress():
    """Get the current initialization progress message"""
    return _initialization_progress


def get_initialization_error():
    """Get the initialization error if any"""
    return _initialization_error


def is_ready():
    """Check if the model is ready for use"""
    return (
        _initialization_state == InitializationState.READY.value and _model is not None
    )


def is_initializing():
    """Check if the model is currently initializing"""
    return _initialization_state == InitializationState.INITIALIZING.value


def is_multilingual():
    """Check if the loaded model supports multilingual generation"""
    return _is_multilingual


def get_supported_languages():
    """Get the dictionary of supported languages"""
    if _supported_languages:
        return _supported_languages.copy()
    return _resolve_supported_languages(
        Config.get_model_source(),
        Config.get_model_class(),
    )


def get_default_language():
    """Get the default generation language"""
    return _model_metadata.get("default_language") or Config.get_default_language()


def supports_language(language_id: str):
    """Check if the model supports a specific language"""
    if not language_id:
        return False
    supported_languages = get_supported_languages() or _resolve_supported_languages(
        Config.get_model_source(),
        Config.get_model_class(),
    )
    return language_id.lower() in supported_languages


def get_model_info() -> Dict[str, Any]:
    """Get comprehensive model information"""
    configured_model_class = (
        _model_metadata.get("model_class") or Config.get_model_class()
    )
    configured_supported_languages = (
        _supported_languages
        or _resolve_supported_languages(
            Config.get_model_source(),
            configured_model_class,
        )
    )
    is_multilingual_model = (
        _is_multilingual
        if _is_multilingual is not None
        else configured_model_class == "multilingual"
    )
    resolved_metadata = {
        **_model_metadata,
        "model_source": _model_metadata.get("model_source")
        or Config.get_model_source(),
        "model_class": configured_model_class,
    }

    return {
        "model_type": "multilingual" if is_multilingual_model else "standard",
        "is_multilingual": is_multilingual_model,
        "supported_languages": configured_supported_languages,
        "language_count": len(configured_supported_languages),
        "default_language": get_default_language(),
        "device": _device,
        "is_ready": is_ready(),
        "initialization_state": _initialization_state,
        **resolved_metadata,
    }
