"""
TTS model initialization and pooled model management.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.tts import ChatterboxTTS
from huggingface_hub import snapshot_download

from app.config import Config, detect_device
from app.core.metrics import (
    observe_model_instance_retired,
    observe_pool_status,
)
from app.core.mtl import SUPPORTED_LANGUAGES
from app.core.observability import get_logger, log_event

logger = get_logger(__name__)

# Backwards-compatible primary model reference.
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
_model_pool: list["ModelSlot"] = []
_available_model_ids: Optional[asyncio.Queue[int]] = None


class InitializationState(Enum):
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"


class ModelPoolError(RuntimeError):
    """Base error for model pool failures."""


class ModelNotReadyError(ModelPoolError):
    """Raised when the model pool cannot serve requests."""


class ModelPoolExhaustedError(ModelPoolError):
    """Raised when no model lease is available within the timeout."""


@dataclass
class ModelSlot:
    instance_id: int
    model: Any
    device: str
    healthy: bool = True
    last_error: Optional[str] = None


@dataclass
class ModelLease:
    instance_id: int
    model: Any
    device: str
    broken: bool = False
    failure_reason: Optional[str] = None
    released: bool = False

    def mark_broken(self, reason: str):
        self.broken = True
        self.failure_reason = reason


def _reset_runtime_state():
    global _model, _device, _initialization_state, _initialization_error
    global _initialization_progress, _is_multilingual, _supported_languages
    global _model_metadata, _model_pool, _available_model_ids

    _model = None
    _device = None
    _initialization_state = InitializationState.NOT_STARTED.value
    _initialization_error = None
    _initialization_progress = ""
    _is_multilingual = None
    _supported_languages = {}
    _model_metadata = {
        "model_source": "default",
        "model_class": None,
        "model_type": None,
        "model_repo_id": None,
        "model_revision": None,
        "model_local_path": None,
        "resolved_model_path": None,
        "default_language": "en",
    }
    _model_pool = []
    _available_model_ids = None
    observe_pool_status(
        {
            "configured_instances": Config.MODEL_INSTANCE_COUNT,
            "healthy_instances": 0,
            "available_instances": 0,
            "busy_instances": 0,
            "unhealthy_instances": 0,
        }
    )


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


def _configure_cpu_loading(device: str):
    if device != "cpu":
        return

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


def _healthy_slot_count() -> int:
    return sum(1 for slot in _model_pool if slot.healthy)


def _available_slot_count() -> int:
    if _available_model_ids is None:
        return 0
    return _available_model_ids.qsize()


def _update_runtime_after_slot_failure(instance_id: int, reason: str):
    global _initialization_state, _initialization_error, _initialization_progress

    healthy_count = _healthy_slot_count()
    _initialization_error = f"Model instance {instance_id} failed: {reason}"
    if healthy_count <= 0:
        _initialization_state = InitializationState.ERROR.value
        _initialization_progress = "No healthy model instances available"
    else:
        _initialization_progress = f"Degraded pool capacity: {healthy_count}/{len(_model_pool)} instances healthy"

    observe_model_instance_retired()
    observe_pool_status(get_pool_status())

    log_event(
        logger,
        logging.ERROR,
        "model_instance_retired",
        model_instance_id=instance_id,
        healthy_instances=healthy_count,
        configured_pool_size=len(_model_pool),
        reason=reason,
    )


async def initialize_model():
    """Initialize the configured pool of Chatterbox TTS models."""
    global _model, _device, _initialization_state, _initialization_error
    global _initialization_progress, _is_multilingual, _supported_languages
    global _model_metadata, _model_pool, _available_model_ids

    try:
        _reset_runtime_state()
        _initialization_state = InitializationState.INITIALIZING.value
        _initialization_progress = "Validating configuration..."

        Config.validate()
        _device = detect_device()
        model_source = Config.get_model_source()
        model_class = Config.get_model_class()
        default_language = Config.get_default_language()

        log_event(
            logger,
            logging.INFO,
            "model_pool_initialization_started",
            device=_device,
            voice_sample_path=Config.VOICE_SAMPLE_PATH,
            model_cache_dir=Config.MODEL_CACHE_DIR,
            model_source=model_source,
            model_class=model_class,
            configured_pool_size=Config.MODEL_INSTANCE_COUNT,
            model_repo_id=Config.MODEL_REPO_ID or None,
            model_local_path=Config.MODEL_LOCAL_PATH,
        )

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
        _configure_cpu_loading(_device)
        observe_pool_status(
            {
                "configured_instances": Config.MODEL_INSTANCE_COUNT,
                "healthy_instances": 0,
                "available_instances": 0,
                "busy_instances": 0,
                "unhealthy_instances": 0,
            }
        )

        loop = asyncio.get_running_loop()
        loaded_slots: list[ModelSlot] = []
        available_ids: asyncio.Queue[int] = asyncio.Queue()
        model_metadata: Optional[Dict[str, Any]] = None

        for instance_id in range(Config.MODEL_INSTANCE_COUNT):
            _initialization_progress = (
                f"Loading TTS model {instance_id + 1}/{Config.MODEL_INSTANCE_COUNT}..."
            )
            log_event(
                logger,
                logging.INFO,
                "model_instance_loading",
                model_instance_id=instance_id,
                device=_device,
                model_source=model_source,
                model_class=model_class,
                configured_pool_size=Config.MODEL_INSTANCE_COUNT,
            )
            model, model_metadata = await loop.run_in_executor(
                None,
                lambda ms=model_source, mc=model_class, dv=_device: _load_model_sync(
                    ms, mc, dv
                ),
            )
            loaded_slots.append(
                ModelSlot(instance_id=instance_id, model=model, device=_device)
            )
            available_ids.put_nowait(instance_id)
            log_event(
                logger,
                logging.INFO,
                "model_instance_loaded",
                model_instance_id=instance_id,
                device=_device,
            )

        _model_pool = loaded_slots
        _available_model_ids = available_ids
        _model = loaded_slots[0].model if loaded_slots else None
        _is_multilingual = model_class == "multilingual"
        _supported_languages = _resolve_supported_languages(model_source, model_class)
        _model_metadata = {
            **(model_metadata or {}),
            "default_language": default_language,
        }

        _initialization_state = InitializationState.READY.value
        _initialization_progress = (
            f"Model pool ready ({len(_model_pool)}/{Config.MODEL_INSTANCE_COUNT})"
        )
        _initialization_error = None
        log_event(
            logger,
            logging.INFO,
            "model_pool_initialized",
            device=_device,
            configured_pool_size=Config.MODEL_INSTANCE_COUNT,
            loaded_instances=len(_model_pool),
            supported_languages=list(_supported_languages.keys()),
            resolved_model_path=_model_metadata.get("resolved_model_path"),
        )
        observe_pool_status(get_pool_status())
        return _model

    except Exception as e:
        _initialization_state = InitializationState.ERROR.value
        _initialization_error = str(e)
        _initialization_progress = f"Failed: {str(e)}"
        _model = None
        _model_pool = []
        _available_model_ids = None
        observe_pool_status(
            {
                "configured_instances": Config.MODEL_INSTANCE_COUNT,
                "healthy_instances": 0,
                "available_instances": 0,
                "busy_instances": 0,
                "unhealthy_instances": 0,
            }
        )
        logger.exception(
            "model_pool_initialization_failed",
            extra={
                "event": "model_pool_initialization_failed",
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
        )
        raise e


async def acquire_model_lease(timeout_seconds: Optional[float] = None) -> ModelLease:
    """Lease one healthy model instance for a full request."""
    if not is_ready() or _available_model_ids is None:
        raise ModelNotReadyError("Model pool not ready")

    wait_seconds = (
        Config.MAX_QUEUE_WAIT_SECONDS if timeout_seconds is None else timeout_seconds
    )
    queue = _available_model_ids

    while True:
        try:
            if wait_seconds <= 0:
                instance_id = queue.get_nowait()
            else:
                instance_id = await asyncio.wait_for(queue.get(), timeout=wait_seconds)
        except asyncio.QueueEmpty as exc:
            raise ModelPoolExhaustedError("No model instances available") from exc
        except asyncio.TimeoutError as exc:
            raise ModelPoolExhaustedError(
                "Timed out waiting for an available model instance"
            ) from exc

        if instance_id >= len(_model_pool):
            continue

        slot = _model_pool[instance_id]
        if not slot.healthy:
            continue

        lease = ModelLease(
            instance_id=slot.instance_id,
            model=slot.model,
            device=slot.device,
        )
        observe_pool_status(get_pool_status())
        return lease


async def release_model_lease(lease: Optional[ModelLease]):
    """Release a model lease or retire the slot if it failed."""
    if lease is None or lease.released:
        return

    lease.released = True
    if lease.instance_id >= len(_model_pool):
        return

    slot = _model_pool[lease.instance_id]
    if lease.broken:
        slot.healthy = False
        slot.last_error = lease.failure_reason
        _update_runtime_after_slot_failure(
            lease.instance_id, lease.failure_reason or ""
        )
        return

    if slot.healthy and _available_model_ids is not None:
        _available_model_ids.put_nowait(lease.instance_id)
    observe_pool_status(get_pool_status())


@asynccontextmanager
async def leased_model(timeout_seconds: Optional[float] = None):
    lease = await acquire_model_lease(timeout_seconds)
    try:
        yield lease
    finally:
        await release_model_lease(lease)


def get_pool_status() -> Dict[str, Any]:
    """Return the current model pool state for health checks."""
    healthy_instances = _healthy_slot_count()
    available_instances = _available_slot_count()
    busy_instances = max(healthy_instances - available_instances, 0)
    unhealthy_instances = max(len(_model_pool) - healthy_instances, 0)
    return {
        "configured_instances": Config.MODEL_INSTANCE_COUNT,
        "loaded_instances": len(_model_pool),
        "healthy_instances": healthy_instances,
        "available_instances": available_instances,
        "busy_instances": busy_instances,
        "unhealthy_instances": unhealthy_instances,
        "ready": is_ready(),
    }


def get_model():
    """Get the primary model instance for compatibility call sites."""
    return _model


def get_device():
    """Get the current device."""
    return _device


def get_initialization_state():
    """Get the current initialization state."""
    return _initialization_state


def get_initialization_progress():
    """Get the current initialization progress message."""
    return _initialization_progress


def get_initialization_error():
    """Get the initialization or latest pool error."""
    return _initialization_error


def is_ready():
    """Check if the model pool can currently accept work."""
    return (
        _initialization_state == InitializationState.READY.value
        and _healthy_slot_count() > 0
        and _available_model_ids is not None
    )


def is_initializing():
    """Check if the model pool is currently initializing."""
    return _initialization_state == InitializationState.INITIALIZING.value


def is_multilingual():
    """Check if the loaded model supports multilingual generation."""
    return _is_multilingual


def get_supported_languages():
    """Get the dictionary of supported languages."""
    if _supported_languages:
        return _supported_languages.copy()
    return _resolve_supported_languages(
        Config.get_model_source(),
        Config.get_model_class(),
    )


def get_default_language():
    """Get the default generation language."""
    return _model_metadata.get("default_language") or Config.get_default_language()


def supports_language(language_id: str):
    """Check if the model supports a specific language."""
    if not language_id:
        return False
    supported_languages = get_supported_languages() or _resolve_supported_languages(
        Config.get_model_source(),
        Config.get_model_class(),
    )
    return language_id.lower() in supported_languages


def get_model_info() -> Dict[str, Any]:
    """Get comprehensive model information."""
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
        "model_instance_count": Config.MODEL_INSTANCE_COUNT,
        "pool_status": get_pool_status(),
        **resolved_metadata,
    }


__all__ = [
    "ModelLease",
    "ModelNotReadyError",
    "ModelPoolExhaustedError",
    "acquire_model_lease",
    "get_default_language",
    "get_device",
    "get_initialization_error",
    "get_initialization_progress",
    "get_initialization_state",
    "get_model",
    "get_model_info",
    "get_pool_status",
    "get_supported_languages",
    "initialize_model",
    "is_initializing",
    "is_multilingual",
    "is_ready",
    "leased_model",
    "release_model_lease",
    "supports_language",
]
