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

from chatterbox.inference import ChatterboxInference
from huggingface_hub import snapshot_download

from app.config import Config, detect_device
from app.core.chatterbox_patches import apply_chatterbox_patches
from app.core.metrics import (
    observe_model_initialization,
    observe_model_instance_load,
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
_reinit_lock: Optional[asyncio.Lock] = None


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


# Retire a slot after this many consecutive non-fatal generation failures in a row.
# A successful request resets the counter. Fatal errors retire the slot immediately.
MAX_CONSECUTIVE_SLOT_FAILURES = 5

# Reinitialize a failed slot up to this many times before giving up permanently.
MAX_SLOT_RECOVERY_ATTEMPTS = 3

# Initial backoff for failure recovery; doubles on each retry (5 → 10 → 20 s).
SLOT_RECOVERY_INITIAL_BACKOFF_SECONDS = 5.0

# Proactively reload a slot after this many requests to clear accumulated
# attention hooks and GPU memory fragmentation.
SLOT_REFRESH_AFTER_REQUESTS = 200


def is_fatal_generation_error(exc: Exception) -> bool:
    """Return True only for errors that leave the model instance in an unrecoverable state.

    Fatal: CUDA OOM, CUDA device errors, NCCL failures — the GPU is in a bad state.
    Non-fatal (transient): everything else — bad input, numerical edge cases, etc.
    """
    try:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
        cuda_error = getattr(torch.cuda, "CudaError", None)
        if cuda_error and isinstance(exc, cuda_error):
            return True
    except Exception:
        pass

    if isinstance(exc, RuntimeError):
        msg = str(exc).upper()
        return "CUDA" in msg or "NCCL" in msg

    return False


@dataclass
class ModelSlot:
    instance_id: int
    model: Any
    device: str
    healthy: bool = True
    last_error: Optional[str] = None
    consecutive_failures: int = 0
    requests_served: int = 0
    reinitializing: bool = False


@dataclass
class ModelLease:
    instance_id: int
    model: Any
    device: str
    broken: bool = False
    soft_failure: bool = False
    failure_reason: Optional[str] = None
    released: bool = False

    def mark_broken(self, reason: str):
        self.broken = True
        self.failure_reason = reason

    def mark_soft_failure(self, reason: str):
        self.soft_failure = True
        self.failure_reason = reason


def _reset_runtime_state():
    global _model, _device, _initialization_state, _initialization_error
    global _initialization_progress, _is_multilingual, _supported_languages
    global _model_metadata, _model_pool, _available_model_ids, _reinit_lock

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
    _reinit_lock = None
    observe_pool_status(
        {
            "configured_instances": Config.MODEL_INSTANCE_COUNT,
            "healthy_instances": 0,
            "available_instances": 0,
            "busy_instances": 0,
            "unhealthy_instances": 0,
        }
    )


def _resolve_supported_languages(model_source: str, model_type: str) -> Dict[str, str]:
    configured_languages = Config.get_configured_supported_languages()
    if configured_languages:
        return configured_languages.copy()
    if model_type == "multilingual" and model_source == "default":
        return SUPPORTED_LANGUAGES.copy()
    return {"en": "English"}


def _load_model_sync(
    model_source: str, model_type: str, device: str
) -> tuple[Any, Dict[str, Any]]:
    apply_chatterbox_patches()
    language = Config.get_default_language()
    metadata: Dict[str, Any] = {
        "model_source": model_source,
        "model_class": model_type,
        "model_type": model_type,
        "model_repo_id": Config.MODEL_REPO_ID or None,
        "model_revision": Config.MODEL_REVISION,
        "model_local_path": Config.MODEL_LOCAL_PATH,
        "resolved_model_path": None,
        "default_language": language,
    }
    inference_kwargs = dict(
        model_type=model_type,
        language=language,
        device=device,
        normalize_text=Config.NORMALIZE_TEXT,
        sentence_split=True,
        inter_sentence_silence_ms=100,
    )

    if model_source == "default":
        model = ChatterboxInference.from_pretrained(**inference_kwargs)
        model.prepare_conditionals(Config.VOICE_SAMPLE_PATH)
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
        model = ChatterboxInference.from_local(ckpt_dir=resolved_model_path, **inference_kwargs)
        model.prepare_conditionals(Config.VOICE_SAMPLE_PATH)
        return model, metadata

    if model_source == "local_dir":
        resolved_model_path = os.path.abspath(Config.MODEL_LOCAL_PATH)
        metadata["model_local_path"] = resolved_model_path
        metadata["resolved_model_path"] = resolved_model_path
        model = ChatterboxInference.from_local(ckpt_dir=resolved_model_path, **inference_kwargs)
        model.prepare_conditionals(Config.VOICE_SAMPLE_PATH)
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

    slot = _model_pool[instance_id]
    if not slot.reinitializing:
        asyncio.create_task(_reinitialize_slot(instance_id, "failure"))


async def _reinitialize_slot(instance_id: int, reason: str) -> None:
    """Reload a single model slot. Used for both failure recovery and scheduled refresh."""
    global _initialization_state

    if _reinit_lock is None:
        return

    async with _reinit_lock:
        slot = _model_pool[instance_id]
        slot.reinitializing = True
        loop = asyncio.get_running_loop()

        log_event(
            logger, logging.INFO, "model_instance_reinit_started",
            model_instance_id=instance_id, reason=reason,
        )

        # Free GPU memory before loading. On a single-GPU deployment both
        # instances share the device; the old model must be released first.
        slot.model = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        is_recovery = reason == "failure"
        max_attempts = MAX_SLOT_RECOVERY_ATTEMPTS if is_recovery else 2
        backoff = SLOT_RECOVERY_INITIAL_BACKOFF_SECONDS if is_recovery else 0.0

        for attempt in range(1, max_attempts + 1):
            if backoff > 0:
                await asyncio.sleep(backoff)
                backoff *= 2

            log_event(
                logger, logging.INFO, "model_instance_reinit_attempt",
                model_instance_id=instance_id, reason=reason, attempt=attempt,
            )

            try:
                model_source = Config.get_model_source()
                model_type = Config.get_model_type()
                new_model, _ = await loop.run_in_executor(
                    None,
                    lambda ms=model_source, mt=model_type, dv=slot.device: (
                        _load_model_sync(ms, mt, dv)
                    ),
                )
            except Exception as exc:
                log_event(
                    logger, logging.WARNING, "model_instance_reinit_attempt_failed",
                    model_instance_id=instance_id, reason=reason,
                    attempt=attempt, error=str(exc),
                )
                if attempt == max_attempts:
                    slot.healthy = False
                    slot.reinitializing = False
                    _update_runtime_after_slot_failure(
                        instance_id,
                        f"reinit exhausted after {attempt} attempts: {exc}",
                    )
                return

            slot.model = new_model
            slot.healthy = True
            slot.last_error = None
            slot.requests_served = 0
            slot.consecutive_failures = 0
            slot.reinitializing = False

            if _available_model_ids is not None:
                _available_model_ids.put_nowait(instance_id)

            if _initialization_state == InitializationState.ERROR.value:
                _initialization_state = InitializationState.READY.value
                _initialization_progress = (
                    f"Pool recovered: {_healthy_slot_count()}/{len(_model_pool)} instances healthy"
                )

            observe_pool_status(get_pool_status())
            log_event(
                logger, logging.INFO, "model_instance_reinit_completed",
                model_instance_id=instance_id, reason=reason,
                healthy_instances=_healthy_slot_count(),
            )
            return


async def initialize_model():
    """Initialize the configured pool of Chatterbox TTS models."""
    global _model, _device, _initialization_state, _initialization_error
    global _initialization_progress, _is_multilingual, _supported_languages
    global _model_metadata, _model_pool, _available_model_ids, _reinit_lock

    overall_started_at = asyncio.get_running_loop().time()
    try:
        _reset_runtime_state()
        _initialization_state = InitializationState.INITIALIZING.value
        _initialization_progress = "Validating configuration..."

        Config.validate()
        _device = detect_device()
        model_source = Config.get_model_source()
        model_type = Config.get_model_type()
        default_language = Config.get_default_language()

        log_event(
            logger,
            logging.INFO,
            "model_pool_initialization_started",
            device=_device,
            voice_sample_path=Config.VOICE_SAMPLE_PATH,
            model_cache_dir=Config.MODEL_CACHE_DIR,
            model_source=model_source,
            model_type=model_type,
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
                model_type=model_type,
                configured_pool_size=Config.MODEL_INSTANCE_COUNT,
            )
            instance_started_at = loop.time()
            try:
                model, model_metadata = await loop.run_in_executor(
                    None,
                    lambda ms=model_source, mt=model_type, dv=_device: (
                        _load_model_sync(ms, mt, dv)
                    ),
                )
            except Exception:
                observe_model_instance_load("error", loop.time() - instance_started_at)
                raise
            observe_model_instance_load("success", loop.time() - instance_started_at)
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
        _reinit_lock = asyncio.Lock()
        _model = loaded_slots[0].model if loaded_slots else None
        _is_multilingual = model_type == "multilingual"  # base and turbo treated as en-only
        _supported_languages = _resolve_supported_languages(model_source, model_type)
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
        observe_model_initialization(
            "success", asyncio.get_running_loop().time() - overall_started_at
        )
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
        observe_model_initialization(
            "error", asyncio.get_running_loop().time() - overall_started_at
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

    if lease.soft_failure:
        slot.consecutive_failures += 1
        slot.last_error = lease.failure_reason
        if slot.consecutive_failures >= MAX_CONSECUTIVE_SLOT_FAILURES:
            slot.healthy = False
            _update_runtime_after_slot_failure(
                lease.instance_id,
                f"retired after {slot.consecutive_failures} consecutive failures: {lease.failure_reason}",
            )
            return
    else:
        slot.consecutive_failures = 0

    slot.requests_served += 1

    if (
        slot.requests_served >= SLOT_REFRESH_AFTER_REQUESTS
        and _reinit_lock is not None
        and not _reinit_lock.locked()
    ):
        slot.reinitializing = True
        asyncio.create_task(_reinitialize_slot(slot.instance_id, "scheduled_refresh"))
        observe_pool_status(get_pool_status())
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
        Config.get_model_type(),
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
        Config.get_model_type(),
    )
    return language_id.lower() in supported_languages


def get_model_info() -> Dict[str, Any]:
    """Get comprehensive model information."""
    configured_model_type = (
        _model_metadata.get("model_type") or Config.get_model_type()
    )
    configured_supported_languages = (
        _supported_languages
        or _resolve_supported_languages(
            Config.get_model_source(),
            configured_model_type,
        )
    )
    is_multilingual_model = (
        _is_multilingual
        if _is_multilingual is not None
        else configured_model_type == "multilingual"
    )
    resolved_metadata = {
        **_model_metadata,
        "model_source": _model_metadata.get("model_source")
        or Config.get_model_source(),
        "model_type": configured_model_type,
    }

    return {
        "model_type": configured_model_type,
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
