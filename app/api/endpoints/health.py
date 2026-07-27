"""
Health check and status endpoints
"""

from fastapi import APIRouter, Response

from app.config import Config
from app.core import get_memory_info
from app.models import HealthResponse
from app.core.tts_model import (
    InitializationState,
    get_model,
    get_device,
    get_model_info,
    get_initialization_state,
    get_initialization_progress,
    get_initialization_error,
    get_pool_status,
    is_ready,
)

base_router = APIRouter()


@base_router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check API health and model status",
)
async def health_check():
    """Health check endpoint - always responds even during initialization"""
    model = get_model()
    device = get_device()
    init_state = get_initialization_state()
    init_progress = get_initialization_progress()
    init_error = get_initialization_error()
    model_info = get_model_info()
    pool_status = get_pool_status()
    ready = is_ready()

    # Determine status based on initialization state
    if init_state == "ready" and ready and pool_status["unhealthy_instances"] == 0:
        status = "healthy"
    elif init_state == "ready" and ready:
        status = "degraded"
    elif init_state == "ready":
        status = "error"
    elif init_state == "initializing":
        status = "initializing"
    elif init_state == "error":
        status = "error"
    else:
        status = "starting"

    return HealthResponse(
        status=status,
        ready=ready,
        model_loaded=model is not None and pool_status["healthy_instances"] > 0,
        device=device or "unknown",
        config={
            "min_text_length": Config.MIN_TEXT_LENGTH,
            "max_total_length": Config.MAX_TOTAL_LENGTH,
            "model_instance_count": Config.MODEL_INSTANCE_COUNT,
            "max_queue_wait_seconds": Config.MAX_QUEUE_WAIT_SECONDS,
            "request_timeout_seconds": Config.REQUEST_TIMEOUT_SECONDS,
            "voice_sample_path": Config.VOICE_SAMPLE_PATH,
            "default_voice_name": Config.DEFAULT_VOICE_NAME,
            "voice_library": Config.get_voice_library(),
            "default_exaggeration": Config.EXAGGERATION,
            "default_cfg_weight": Config.CFG_WEIGHT,
            "default_temperature": Config.TEMPERATURE,
            "model_source": model_info.get("model_source"),
            "model_class": model_info.get("model_class"),
            "model_repo_id": model_info.get("model_repo_id"),
            "model_revision": model_info.get("model_revision"),
            "resolved_model_path": model_info.get("resolved_model_path"),
            "default_language": model_info.get("default_language"),
            "supported_languages": model_info.get("supported_languages", {}),
        },
        pool_status=pool_status,
        memory_info=get_memory_info(),
        initialization_state=init_state,
        initialization_progress=init_progress,
        initialization_error=init_error,
    )


@base_router.get(
    "/healthz/live",
    summary="Liveness probe",
    description=(
        "Returns 200 while the pod should keep running. "
        "Returns 503 when the model pool has permanently failed and the pod should be restarted."
    ),
)
async def liveness_probe(response: Response):
    if get_initialization_state() == InitializationState.ERROR.value:
        from app.core import tts_model as _tts_model
        recovering = any(s.reinitializing for s in _tts_model._model_pool)
        if not recovering:
            response.status_code = 503
            return {"status": "dead", "reason": "model pool permanently failed"}
    return {"status": "alive"}


@base_router.get(
    "/healthz/ready",
    summary="Readiness probe",
    description=(
        "Returns 200 when the pod is ready to serve TTS requests. "
        "Returns 503 during initialisation or when no healthy model instances are available."
    ),
)
async def readiness_probe(response: Response):
    if not is_ready():
        response.status_code = 503
        return {"status": "not_ready", "initialization_state": get_initialization_state()}
    return {"status": "ready"}


@base_router.get(
    "/ping",
    summary="Simple connectivity check",
    description="Basic connectivity test - always responds immediately",
)
async def ping():
    """Simple ping endpoint for connectivity testing"""
    return {"status": "ok", "message": "Server is running"}


# Export the base router for the main app to use
__all__ = ["base_router"]
