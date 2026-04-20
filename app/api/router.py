"""Main API router combining the active endpoints."""

from fastapi import APIRouter

from app.api.endpoints import health, metrics, models, speech

api_router = APIRouter()

api_router.include_router(speech.base_router, tags=["Text-to-Speech"])
api_router.include_router(health.base_router, tags=["Health"])
api_router.include_router(metrics.base_router, tags=["Metrics"])
api_router.include_router(models.base_router, tags=["Models"])
