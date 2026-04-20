"""Prometheus metrics helpers for the API runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    GCCollector,
    Gauge,
    Histogram,
    PlatformCollector,
    ProcessCollector,
    generate_latest,
)


REQUEST_DURATION_SECONDS_BUCKETS = (
    0.1,
    0.25,
    0.5,
    1,
    2.5,
    5,
    10,
    15,
    20,
    30,
    45,
    60,
    90,
    120,
)
LEASE_WAIT_SECONDS_BUCKETS = (0.01, 0.1, 0.5, 1, 2.5, 5, 10, 15, 20, 30, 45, 60)
GENERATION_DURATION_SECONDS_BUCKETS = (0.5, 1, 2.5, 5, 10, 15, 20, 30, 45, 60, 90, 120)
INPUT_CHARS_BUCKETS = (1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000)
AUDIO_SECONDS_BUCKETS = (1, 5, 10, 30, 60, 120, 300, 600)
CHUNK_COUNT_BUCKETS = (1, 2, 5, 10, 20, 50, 100)
FIRST_CHUNK_SECONDS_BUCKETS = (
    0.1,
    0.25,
    0.5,
    1,
    2.5,
    5,
    10,
    15,
    20,
    30,
    45,
    60,
    90,
    120,
)
MODEL_INIT_SECONDS_BUCKETS = (0.5, 1, 2.5, 5, 10, 20, 30, 60, 120, 300)


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
    time_to_first_chunk_seconds: Histogram
    input_chars: Histogram
    audio_seconds: Histogram
    chunk_count: Histogram
    request_failures_total: Counter
    requests_in_progress: Gauge
    requests_waiting_for_lease: Gauge
    model_initialization_seconds: Histogram
    model_instance_load_seconds: Histogram
    pool_configured: Gauge
    pool_healthy: Gauge
    pool_available: Gauge
    pool_busy: Gauge
    pool_unhealthy: Gauge
    cpu_memory_mb: Gauge
    cpu_memory_percent: Gauge
    gpu_memory_allocated_mb: Gauge
    gpu_memory_reserved_mb: Gauge
    gpu_memory_max_allocated_mb: Gauge


def _build_metrics_state() -> MetricsState:
    registry = CollectorRegistry()
    ProcessCollector(registry=registry)
    PlatformCollector(registry=registry)
    GCCollector(registry=registry)

    requests_waiting_for_lease = Gauge(
        "chatterbox_tts_requests_waiting_for_lease",
        "Requests currently waiting for a model lease",
        registry=registry,
    )
    requests_waiting_for_lease.set(0)

    cpu_memory_mb = Gauge(
        "chatterbox_tts_cpu_memory_mb",
        "Process resident CPU memory in megabytes",
        registry=registry,
    )
    cpu_memory_percent = Gauge(
        "chatterbox_tts_cpu_memory_percent",
        "Process CPU memory percent",
        registry=registry,
    )
    gpu_memory_allocated_mb = Gauge(
        "chatterbox_tts_gpu_memory_allocated_mb",
        "Current CUDA allocated memory in megabytes",
        registry=registry,
    )
    gpu_memory_reserved_mb = Gauge(
        "chatterbox_tts_gpu_memory_reserved_mb",
        "Current CUDA reserved memory in megabytes",
        registry=registry,
    )
    gpu_memory_max_allocated_mb = Gauge(
        "chatterbox_tts_gpu_memory_max_allocated_mb",
        "Peak CUDA allocated memory in megabytes",
        registry=registry,
    )

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
            buckets=REQUEST_DURATION_SECONDS_BUCKETS,
            registry=registry,
        ),
        lease_wait_seconds=Histogram(
            "chatterbox_tts_lease_wait_seconds",
            "Time spent waiting for a model lease",
            ["route", "mode"],
            buckets=LEASE_WAIT_SECONDS_BUCKETS,
            registry=registry,
        ),
        generation_duration_seconds=Histogram(
            "chatterbox_tts_generation_duration_seconds",
            "Time spent generating audio after lease acquisition",
            ["route", "mode", "outcome"],
            buckets=GENERATION_DURATION_SECONDS_BUCKETS,
            registry=registry,
        ),
        time_to_first_chunk_seconds=Histogram(
            "chatterbox_tts_time_to_first_chunk_seconds",
            "Time from request start until first SSE audio chunk is emitted",
            ["route"],
            buckets=FIRST_CHUNK_SECONDS_BUCKETS,
            registry=registry,
        ),
        input_chars=Histogram(
            "chatterbox_tts_input_chars",
            "Input text length in characters",
            ["route", "mode"],
            buckets=INPUT_CHARS_BUCKETS,
            registry=registry,
        ),
        audio_seconds=Histogram(
            "chatterbox_tts_audio_seconds",
            "Generated audio length in seconds",
            ["route", "mode"],
            buckets=AUDIO_SECONDS_BUCKETS,
            registry=registry,
        ),
        chunk_count=Histogram(
            "chatterbox_tts_chunk_count",
            "Chunk count per request",
            ["route", "mode"],
            buckets=CHUNK_COUNT_BUCKETS,
            registry=registry,
        ),
        request_failures_total=Counter(
            "chatterbox_tts_request_failures_total",
            "Request failures by reason, stage, and mode",
            ["reason", "stage", "mode"],
            registry=registry,
        ),
        requests_in_progress=Gauge(
            "chatterbox_tts_requests_in_progress",
            "Requests currently being processed",
            ["route", "mode"],
            registry=registry,
        ),
        requests_waiting_for_lease=requests_waiting_for_lease,
        model_initialization_seconds=Histogram(
            "chatterbox_tts_model_initialization_seconds",
            "Total model pool initialization time in seconds",
            ["outcome"],
            buckets=MODEL_INIT_SECONDS_BUCKETS,
            registry=registry,
        ),
        model_instance_load_seconds=Histogram(
            "chatterbox_tts_model_instance_load_seconds",
            "Time to load a single model instance in seconds",
            ["outcome"],
            buckets=MODEL_INIT_SECONDS_BUCKETS,
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
        cpu_memory_mb=cpu_memory_mb,
        cpu_memory_percent=cpu_memory_percent,
        gpu_memory_allocated_mb=gpu_memory_allocated_mb,
        gpu_memory_reserved_mb=gpu_memory_reserved_mb,
        gpu_memory_max_allocated_mb=gpu_memory_max_allocated_mb,
    )


_metrics = _build_metrics_state()


def get_registry() -> CollectorRegistry:
    return _metrics.registry


def _update_memory_gauges():
    from app.core.memory import get_memory_info

    memory_info = get_memory_info()
    _metrics.cpu_memory_mb.set(memory_info.get("cpu_memory_mb", 0.0))
    _metrics.cpu_memory_percent.set(memory_info.get("cpu_memory_percent", 0.0))
    _metrics.gpu_memory_allocated_mb.set(
        memory_info.get("gpu_memory_allocated_mb", 0.0)
    )
    _metrics.gpu_memory_reserved_mb.set(memory_info.get("gpu_memory_reserved_mb", 0.0))
    _metrics.gpu_memory_max_allocated_mb.set(
        memory_info.get("gpu_memory_max_allocated_mb", 0.0)
    )


def render_metrics() -> tuple[bytes, str]:
    _update_memory_gauges()
    return generate_latest(_metrics.registry), CONTENT_TYPE_LATEST


def reset_metrics_for_tests():
    global _metrics
    _metrics = _build_metrics_state()


def observe_request_started(route: str, mode: str, input_chars: int):
    _metrics.requests_in_progress.labels(route=route, mode=mode).inc()
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
    _metrics.requests_in_progress.labels(route=route, mode=mode).dec()
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


def observe_requests_waiting_for_lease(delta: int):
    if delta >= 0:
        _metrics.requests_waiting_for_lease.inc(delta)
    else:
        _metrics.requests_waiting_for_lease.dec(-delta)


def observe_time_to_first_chunk(route: str, elapsed_seconds: float):
    _metrics.time_to_first_chunk_seconds.labels(route=route).observe(
        max(elapsed_seconds, 0.0)
    )


def observe_request_failure(reason: str, stage: str, mode: str):
    _metrics.request_failures_total.labels(reason=reason, stage=stage, mode=mode).inc()


def observe_model_initialization(outcome: str, elapsed_seconds: float):
    _metrics.model_initialization_seconds.labels(outcome=outcome).observe(
        max(elapsed_seconds, 0.0)
    )


def observe_model_instance_load(outcome: str, elapsed_seconds: float):
    _metrics.model_instance_load_seconds.labels(outcome=outcome).observe(
        max(elapsed_seconds, 0.0)
    )


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
    "observe_model_initialization",
    "observe_model_instance_load",
    "observe_lease_acquire_failure",
    "observe_model_instance_retired",
    "observe_pool_status",
    "observe_request_failure",
    "observe_request_finished",
    "observe_request_started",
    "observe_requests_waiting_for_lease",
    "observe_time_to_first_chunk",
    "render_metrics",
    "reset_metrics_for_tests",
]
