"""Prometheus metrics endpoint."""

from fastapi import APIRouter, Response

from app.core.metrics import render_metrics

base_router = APIRouter()


@base_router.get("/metrics", summary="Prometheus metrics")
async def metrics():
    payload, content_type = render_metrics()
    return Response(content=payload, media_type=content_type)


__all__ = ["base_router", "metrics"]
