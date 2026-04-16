"""Prometheus metrics helpers for the API runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


@dataclass
class MetricsState:
    registry: CollectorRegistry
    requests_total: Counter
    sse_disconnects_total: Counter
    lease_acquire_failures_total: Counter
    model_instance_retirements_total: Counter
    request_duration_seconds: Histogram
    lease_wait_seconds: Histogram
    generation_duration_seconds: Histogram
    input_chars: Histogram
    audio_seconds: Histogram
    chunk_count: Histogram
    pool_configured: Gauge
    pool_healthy: Gauge
    pool_available: Gauge
    pool_busy: Gauge
    pool_unhealthy: Gauge


def _build_metrics_state() -> MetricsState:
    registry = CollectorRegistry()
    return MetricsState(
        registry=registry,
        requests_total=Counter(
            "chatterbox_tts_requests_total",
            "Completed or terminal requests by route, mode, and outcome",
            ["route", "mode", "outcome"],
            registry=registry,
        ),
        sse_disconnects_total=Counter(
            "chatterbox_tts_sse_disconnects_total",
            "SSE requests terminated due to client disconnect",
            ["route"],
            registry=registry,
        ),
        lease_acquire_failures_total=Counter(
            "chatterbox_tts_lease_acquire_failures_total",
            "Lease acquisition failures by reason",
            ["reason"],
            registry=registry,
        ),
        model_instance_retirements_total=Counter(
            "chatterbox_tts_model_instance_retirements_total",
            "Model instances retired from the pool",
            registry=registry,
        ),
        request_duration_seconds=Histogram(
            "chatterbox_tts_request_duration_seconds",
            "Total request duration in seconds",
            ["route", "mode", "outcome"],
            registry=registry,
        ),
        lease_wait_seconds=Histogram(
            "chatterbox_tts_lease_wait_seconds",
            "Time spent waiting for a model lease",
            ["route", "mode"],
            registry=registry,
        ),
        generation_duration_seconds=Histogram(
            "chatterbox_tts_generation_duration_seconds",
            "Time spent generating audio after lease acquisition",
            ["route", "mode", "outcome"],
            registry=registry,
        ),
        input_chars=Histogram(
            "chatterbox_tts_input_chars",
            "Input text length in characters",
            ["route", "mode"],
            registry=registry,
        ),
        audio_seconds=Histogram(
            "chatterbox_tts_audio_seconds",
            "Generated audio length in seconds",
            ["route", "mode"],
            registry=registry,
        ),
        chunk_count=Histogram(
            "chatterbox_tts_chunk_count",
            "Chunk count per request",
            ["route", "mode"],
            registry=registry,
        ),
        pool_configured=Gauge(
            "chatterbox_tts_pool_configured_instances",
            "Configured model instances",
            registry=registry,
        ),
        pool_healthy=Gauge(
            "chatterbox_tts_pool_healthy_instances",
            "Healthy model instances",
            registry=registry,
        ),
        pool_available=Gauge(
            "chatterbox_tts_pool_available_instances",
            "Available model instances",
            registry=registry,
        ),
        pool_busy=Gauge(
            "chatterbox_tts_pool_busy_instances",
            "Busy model instances",
            registry=registry,
        ),
        pool_unhealthy=Gauge(
            "chatterbox_tts_pool_unhealthy_instances",
            "Unhealthy model instances",
            registry=registry,
        ),
    )


_metrics = _build_metrics_state()


def get_registry() -> CollectorRegistry:
    return _metrics.registry


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(_metrics.registry), CONTENT_TYPE_LATEST


def reset_metrics_for_tests():
    global _metrics
    _metrics = _build_metrics_state()


def observe_request_started(route: str, mode: str, input_chars: int):
    _metrics.input_chars.labels(route=route, mode=mode).observe(input_chars)


def observe_request_finished(
    route: str,
    mode: str,
    outcome: str,
    elapsed_seconds: float,
    lease_wait_seconds: Optional[float] = None,
    generation_duration_seconds: Optional[float] = None,
    audio_seconds: Optional[float] = None,
    chunk_count: Optional[int] = None,
):
    _metrics.requests_total.labels(route=route, mode=mode, outcome=outcome).inc()
    _metrics.request_duration_seconds.labels(
        route=route, mode=mode, outcome=outcome
    ).observe(max(elapsed_seconds, 0.0))

    if lease_wait_seconds is not None:
        _metrics.lease_wait_seconds.labels(route=route, mode=mode).observe(
            max(lease_wait_seconds, 0.0)
        )
    if generation_duration_seconds is not None:
        _metrics.generation_duration_seconds.labels(
            route=route, mode=mode, outcome=outcome
        ).observe(max(generation_duration_seconds, 0.0))
    if audio_seconds is not None:
        _metrics.audio_seconds.labels(route=route, mode=mode).observe(
            max(audio_seconds, 0.0)
        )
    if chunk_count is not None:
        _metrics.chunk_count.labels(route=route, mode=mode).observe(max(chunk_count, 0))

    if outcome == "disconnect":
        _metrics.sse_disconnects_total.labels(route=route).inc()


def observe_lease_acquire_failure(reason: str):
    _metrics.lease_acquire_failures_total.labels(reason=reason).inc()


def observe_model_instance_retired():
    _metrics.model_instance_retirements_total.inc()


def observe_pool_status(pool_status: dict):
    _metrics.pool_configured.set(pool_status.get("configured_instances", 0))
    _metrics.pool_healthy.set(pool_status.get("healthy_instances", 0))
    _metrics.pool_available.set(pool_status.get("available_instances", 0))
    _metrics.pool_busy.set(pool_status.get("busy_instances", 0))
    _metrics.pool_unhealthy.set(pool_status.get("unhealthy_instances", 0))


__all__ = [
    "get_registry",
    "observe_lease_acquire_failure",
    "observe_model_instance_retired",
    "observe_pool_status",
    "observe_request_finished",
    "observe_request_started",
    "render_metrics",
    "reset_metrics_for_tests",
]
