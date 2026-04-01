"""
Model listing endpoints (OpenAI compatibility)
"""

from fastapi import APIRouter

from app.models import ModelsResponse, ModelInfo
from app.core import add_route_aliases
from app.core.tts_model import get_model_info

# Create router with aliasing support
base_router = APIRouter()
router = add_route_aliases(base_router)


@router.get(
    "/models",
    response_model=ModelsResponse,
    summary="List models",
    description="List available models (OpenAI API compatibility)",
)
async def list_models():
    """List available models (OpenAI API compatibility)"""
    model_info = get_model_info()
    model_id = (
        model_info.get("model_repo_id")
        or model_info.get("resolved_model_path")
        or (
            "chatterbox-multilingual"
            if model_info.get("is_multilingual")
            else "chatterbox-tts-1"
        )
    )
    if model_info.get("model_repo_id"):
        owned_by = model_info["model_repo_id"].split("/", 1)[0]
    elif model_info.get("model_source") == "local_dir":
        owned_by = "local"
    else:
        owned_by = "resemble-ai"

    return ModelsResponse(
        object="list",
        data=[
            ModelInfo(
                id=model_id, object="model", created=1677649963, owned_by=owned_by
            )
        ],
    )


# Export the base router for the main app to use
__all__ = ["base_router"]
