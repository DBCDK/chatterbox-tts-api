"""Unit tests for Prometheus metrics integration."""

import asyncio
from itertools import count

import pytest
import torch
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.api.endpoints import metrics as metrics_endpoint
from app.api.endpoints import speech
from app.config import Config
from app.core.metrics import render_metrics, reset_metrics_for_tests
import app.core.tts_model as tts_model
from app.models import TTSRequest


class RecordingModel:
    def __init__(self, name: str):
        self.name = name
        self.sr = 24000
        self.generated_texts: list[str] = []

    def generate(self, **kwargs):
        self.generated_texts.append(kwargs["text"])
        return torch.zeros(1, 128)


class FakeRequest:
    def __init__(self, disconnected_sequence=None):
        self._sequence = list(disconnected_sequence or [])
        self._last = self._sequence[-1] if self._sequence else False

    async def is_disconnected(self):
        if self._sequence:
            self._last = self._sequence.pop(0)
        return self._last


@pytest.fixture(autouse=True)
def reset_runtime_state():
    reset_metrics_for_tests()
    tts_model._reset_runtime_state()
    yield
    reset_metrics_for_tests()
    tts_model._reset_runtime_state()


@pytest.fixture(scope="session", autouse=True)
def check_api_health():
    return None


def _configure_test_pool(monkeypatch, pool_size: int, model_factory=RecordingModel):
    model_ids = count()

    def fake_load_model_sync(model_source: str, model_class: str, device: str):
        instance_id = next(model_ids)
        return model_factory(f"model-{instance_id}"), {
            "model_source": model_source,
            "model_class": model_class,
            "model_type": model_class,
            "model_repo_id": None,
            "model_revision": None,
            "model_local_path": None,
            "resolved_model_path": None,
            "default_language": "en",
        }

    monkeypatch.setattr(Config, "MODEL_INSTANCE_COUNT", pool_size)
    monkeypatch.setattr(Config, "MAX_QUEUE_WAIT_SECONDS", 0)
    monkeypatch.setattr(Config, "REQUEST_TIMEOUT_SECONDS", 120)
    monkeypatch.setattr(tts_model, "detect_device", lambda: "cpu")
    monkeypatch.setattr(tts_model, "_configure_cpu_loading", lambda device: None)
    monkeypatch.setattr(tts_model, "_load_model_sync", fake_load_model_sync)


def _metrics_text() -> str:
    payload, _ = render_metrics()
    return payload.decode("utf-8")


def _has_metric_line(metrics_text: str, metric_name: str, *fragments: str) -> bool:
    for line in metrics_text.splitlines():
        if metric_name not in line:
            continue
        if all(fragment in line for fragment in fragments):
            return True
    return False


def test_metrics_endpoint_returns_prometheus_payload():
    app = FastAPI()
    app.include_router(metrics_endpoint.base_router)
    client = TestClient(app)

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "chatterbox_tts_requests_total" in response.text


def test_successful_audio_request_updates_metrics(monkeypatch):
    _configure_test_pool(monkeypatch, pool_size=1)
    monkeypatch.setattr(
        speech.ta,
        "save",
        lambda buffer, audio, sample_rate, format: buffer.write(b"wav"),
    )

    async def scenario():
        await tts_model.initialize_model()
        await speech.text_to_speech(TTSRequest(input="hello there"), FakeRequest())

    asyncio.run(scenario())

    metrics_text = _metrics_text()
    assert _has_metric_line(
        metrics_text,
        "chatterbox_tts_requests_total",
        'route="/v1/audio/speech"',
        'mode="audio"',
        'outcome="success"',
        "1.0",
    )
    assert "chatterbox_tts_pool_available_instances 1.0" in metrics_text
    assert "chatterbox_tts_audio_seconds_bucket{" in metrics_text
    assert (
        'chatterbox_tts_input_chars_bucket{le="1.0",mode="audio",route="/v1/audio/speech"} 0.0'
        in metrics_text
    )
    assert (
        'chatterbox_tts_audio_seconds_bucket{le="1.0",mode="audio",route="/v1/audio/speech"}'
        in metrics_text
    )
    assert (
        'chatterbox_tts_request_duration_seconds_bucket{le="15.0",mode="audio",outcome="success",route="/v1/audio/speech"}'
        in metrics_text
    )
    assert (
        'chatterbox_tts_request_duration_seconds_bucket{le="0.005",mode="audio",outcome="success",route="/v1/audio/speech"}'
        not in metrics_text
    )
    assert (
        'chatterbox_tts_lease_wait_seconds_bucket{le="15.0",mode="audio",route="/v1/audio/speech"}'
        in metrics_text
    )
    assert (
        'chatterbox_tts_lease_wait_seconds_bucket{le="0.005",mode="audio",route="/v1/audio/speech"}'
        not in metrics_text
    )
    assert (
        'chatterbox_tts_generation_duration_seconds_bucket{le="15.0",mode="audio",outcome="success",route="/v1/audio/speech"}'
        in metrics_text
    )
    assert (
        'chatterbox_tts_generation_duration_seconds_bucket{le="0.005",mode="audio",outcome="success",route="/v1/audio/speech"}'
        not in metrics_text
    )
    assert (
        'chatterbox_tts_audio_seconds_bucket{le="0.005",mode="audio",route="/v1/audio/speech"}'
        not in metrics_text
    )


def test_overload_updates_request_and_lease_failure_metrics(monkeypatch):
    _configure_test_pool(monkeypatch, pool_size=1)

    async def scenario():
        await tts_model.initialize_model()
        held_lease = await tts_model.acquire_model_lease(0)

        with pytest.raises(HTTPException) as exc_info:
            await speech.text_to_speech(TTSRequest(input="hello there"), FakeRequest())

        assert exc_info.value.status_code == 503
        await tts_model.release_model_lease(held_lease)

    asyncio.run(scenario())

    metrics_text = _metrics_text()
    assert _has_metric_line(
        metrics_text,
        "chatterbox_tts_requests_total",
        'route="/v1/audio/speech"',
        'mode="audio"',
        'outcome="overload"',
        "1.0",
    )
    assert _has_metric_line(
        metrics_text,
        "chatterbox_tts_lease_acquire_failures_total",
        'reason="no_capacity"',
        "1.0",
    )


def test_sse_disconnect_updates_disconnect_metrics(monkeypatch):
    _configure_test_pool(monkeypatch, pool_size=1)

    async def scenario():
        await tts_model.initialize_model()
        context = speech._new_request_context(
            mode="sse",
            client_request=FakeRequest([False, False, False, True, True]),
        )
        lease = await speech._acquire_request_lease(context)

        async for _ in speech.generate_speech_sse(
            context=context,
            lease=lease,
            text="Sentence one. Sentence two. Sentence three.",
            voice_sample_path=Config.VOICE_SAMPLE_PATH,
            language_id=None,
            exaggeration=None,
            cfg_weight=None,
            temperature=None,
            streaming_chunk_size=20,
            streaming_strategy="sentence",
            streaming_quality="balanced",
        ):
            pass

    asyncio.run(scenario())

    metrics_text = _metrics_text()
    assert _has_metric_line(
        metrics_text,
        "chatterbox_tts_requests_total",
        'route="/v1/audio/speech"',
        'mode="sse"',
        'outcome="disconnect"',
        "1.0",
    )
    assert _has_metric_line(
        metrics_text,
        "chatterbox_tts_sse_disconnects_total",
        'route="/v1/audio/speech"',
        "1.0",
    )
    assert (
        'chatterbox_tts_chunk_count_bucket{le="1.0",mode="sse",route="/v1/audio/speech"}'
        in metrics_text
    )
    assert (
        'chatterbox_tts_chunk_count_bucket{le="0.005",mode="sse",route="/v1/audio/speech"}'
        not in metrics_text
    )
