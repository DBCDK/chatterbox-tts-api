"""Unit tests for Phase 2 timeout and disconnect behavior."""

import asyncio
import time
from itertools import count

import pytest
import torch
from fastapi import HTTPException

from app.api.endpoints import speech
from app.config import Config
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


class DelayedModel(RecordingModel):
    def generate(self, **kwargs):
        time.sleep(0.03)
        return super().generate(**kwargs)


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
    tts_model._reset_runtime_state()
    yield
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
    monkeypatch.setattr(Config, "MAX_QUEUE_WAIT_SECONDS", 10)
    monkeypatch.setattr(Config, "REQUEST_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(tts_model, "detect_device", lambda: "cpu")
    monkeypatch.setattr(tts_model, "_configure_cpu_loading", lambda device: None)
    monkeypatch.setattr(tts_model, "_load_model_sync", fake_load_model_sync)


def test_non_streaming_timeout_returns_504_and_releases_lease(monkeypatch):
    _configure_test_pool(monkeypatch, pool_size=1, model_factory=DelayedModel)
    monkeypatch.setattr(
        speech.ta,
        "save",
        lambda buffer, audio, sample_rate, format: buffer.write(b"wav"),
    )
    long_text = ("Sentence one. Sentence two. Sentence three. " * 12).strip()

    async def scenario():
        await tts_model.initialize_model()

        with pytest.raises(HTTPException) as exc_info:
            await speech.text_to_speech(
                TTSRequest(input=long_text),
                FakeRequest(),
            )

        assert exc_info.value.status_code == 504
        pool_status = tts_model.get_pool_status()
        assert pool_status["healthy_instances"] == 1
        assert pool_status["available_instances"] == 1

    asyncio.run(scenario())


def test_timeout_during_lease_wait_uses_request_timeout(monkeypatch):
    _configure_test_pool(monkeypatch, pool_size=1)
    monkeypatch.setattr(Config, "REQUEST_TIMEOUT_SECONDS", 0.01)

    async def scenario():
        await tts_model.initialize_model()
        held_lease = await tts_model.acquire_model_lease(0)
        context = speech._new_request_context(
            mode="audio", client_request=FakeRequest()
        )

        with pytest.raises(speech.RequestTimeoutExceeded):
            await speech._acquire_request_lease(context)

        await tts_model.release_model_lease(held_lease)
        pool_status = tts_model.get_pool_status()
        assert pool_status["healthy_instances"] == 1
        assert pool_status["available_instances"] == 1

    asyncio.run(scenario())


def test_sse_timeout_stops_before_done_and_keeps_model_healthy(monkeypatch):
    _configure_test_pool(monkeypatch, pool_size=1, model_factory=DelayedModel)
    long_text = ("Sentence one. Sentence two. Sentence three. " * 12).strip()

    async def scenario():
        await tts_model.initialize_model()
        context = speech._new_request_context(mode="sse", client_request=FakeRequest())
        lease = await speech._acquire_request_lease(context)

        events = []
        async for event in speech.generate_speech_sse(
            context=context,
            lease=lease,
            text=long_text,
            voice_sample_path=Config.VOICE_SAMPLE_PATH,
            language_id=None,
            exaggeration=None,
            cfg_weight=None,
            temperature=None,
            streaming_chunk_size=20,
            streaming_strategy="sentence",
            streaming_quality="balanced",
        ):
            events.append(event)

        assert any("speech.audio.info" in event for event in events)
        assert not any("speech.audio.done" in event for event in events)
        pool_status = tts_model.get_pool_status()
        assert pool_status["healthy_instances"] == 1
        assert pool_status["available_instances"] == 1

    asyncio.run(scenario())


def test_sse_disconnect_stops_scheduling_new_chunks(monkeypatch):
    _configure_test_pool(monkeypatch, pool_size=1)
    disconnecting_request = FakeRequest([False, False, False, True, True, True])

    async def scenario():
        await tts_model.initialize_model()
        context = speech._new_request_context(
            mode="sse",
            client_request=disconnecting_request,
        )
        lease = await speech._acquire_request_lease(context)

        events = []
        async for event in speech.generate_speech_sse(
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
            events.append(event)

        model = tts_model._model_pool[0].model
        assert len(model.generated_texts) == 1
        assert not any("speech.audio.done" in event for event in events)
        assert tts_model.get_pool_status()["available_instances"] == 1

    asyncio.run(scenario())
