"""
Health check and status endpoints
"""

from fastapi import APIRouter

from app.config import Config
from app.core import get_memory_info
from app.models import HealthResponse
from app.core.tts_model import (
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
            "max_chunk_length": Config.MAX_CHUNK_LENGTH,
            "max_total_length": Config.MAX_TOTAL_LENGTH,
            "model_instance_count": Config.MODEL_INSTANCE_COUNT,
            "max_queue_wait_seconds": Config.MAX_QUEUE_WAIT_SECONDS,
            "request_timeout_seconds": Config.REQUEST_TIMEOUT_SECONDS,
            "voice_sample_path": Config.VOICE_SAMPLE_PATH,
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
    "/ping",
    summary="Simple connectivity check",
    description="Basic connectivity test - always responds immediately",
)
async def ping():
    """Simple ping endpoint for connectivity testing"""
    return {"status": "ok", "message": "Server is running"}


# Export the base router for the main app to use
__all__ = ["base_router"]
