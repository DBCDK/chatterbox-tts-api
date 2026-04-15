"""Unit tests for Phase 1 model pooling behavior."""

import asyncio
from itertools import count

import pytest
import torch

from app.api.endpoints import speech
from app.config import Config
import app.core.tts_model as tts_model


class RecordingModel:
    def __init__(self, name: str):
        self.name = name
        self.sr = 24000
        self.generated_texts: list[str] = []

    def generate(self, **kwargs):
        self.generated_texts.append(kwargs["text"])
        return torch.zeros(1, 128)


class FailingModel(RecordingModel):
    def generate(self, **kwargs):
        raise RuntimeError(f"generation failed in {self.name}")


@pytest.fixture(autouse=True)
def reset_runtime_state():
    tts_model._reset_runtime_state()
    yield
    tts_model._reset_runtime_state()


@pytest.fixture(scope="session", autouse=True)
def check_api_health():
    """Override the integration-test health gate for pure unit tests."""
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
    monkeypatch.setattr(tts_model, "detect_device", lambda: "cpu")
    monkeypatch.setattr(tts_model, "_configure_cpu_loading", lambda device: None)
    monkeypatch.setattr(tts_model, "_load_model_sync", fake_load_model_sync)


def test_model_pool_limits_parallel_leases(monkeypatch):
    _configure_test_pool(monkeypatch, pool_size=2)

    async def scenario():
        await tts_model.initialize_model()

        lease_one = await tts_model.acquire_model_lease(0)
        lease_two = await tts_model.acquire_model_lease(0)

        assert lease_one.instance_id != lease_two.instance_id

        with pytest.raises(tts_model.ModelPoolExhaustedError):
            await tts_model.acquire_model_lease(0)

        await tts_model.release_model_lease(lease_one)
        await tts_model.release_model_lease(lease_two)

        pool_status = tts_model.get_pool_status()
        assert pool_status["available_instances"] == 2
        assert pool_status["healthy_instances"] == 2

    asyncio.run(scenario())


def test_request_failure_releases_healthy_lease(monkeypatch):
    _configure_test_pool(monkeypatch, pool_size=1)
    monkeypatch.setattr(
        speech.ta,
        "save",
        lambda buffer, audio, sample_rate, format: (_ for _ in ()).throw(
            RuntimeError("wav write failed")
        ),
    )

    async def scenario():
        await tts_model.initialize_model()
        lease = await tts_model.acquire_model_lease(0)

        with pytest.raises(RuntimeError, match="wav write failed"):
            await speech._generate_full_audio(
                lease=lease,
                text="Sentence one. Sentence two.",
                voice_sample_path=Config.VOICE_SAMPLE_PATH,
                language_id=None,
                exaggeration=None,
                cfg_weight=None,
                temperature=None,
            )

        await tts_model.release_model_lease(lease)

        pool_status = tts_model.get_pool_status()
        assert pool_status["healthy_instances"] == 1
        assert pool_status["available_instances"] == 1
        assert tts_model.is_ready() is True

    asyncio.run(scenario())


def test_broken_model_instance_is_retired(monkeypatch):
    _configure_test_pool(monkeypatch, pool_size=1, model_factory=FailingModel)

    async def scenario():
        await tts_model.initialize_model()
        lease = await tts_model.acquire_model_lease(0)

        with pytest.raises(RuntimeError, match="generation failed"):
            await speech._generate_chunk_audio(
                lease=lease,
                chunk="hello",
                voice_sample_path=Config.VOICE_SAMPLE_PATH,
                language_id=None,
                exaggeration=None,
                cfg_weight=None,
                temperature=None,
            )

        await tts_model.release_model_lease(lease)

        pool_status = tts_model.get_pool_status()
        assert pool_status["healthy_instances"] == 0
        assert pool_status["unhealthy_instances"] == 1
        assert pool_status["available_instances"] == 0
        assert tts_model.is_ready() is False
        assert tts_model.get_initialization_state() == "error"

    asyncio.run(scenario())


def test_non_streaming_request_keeps_one_stable_model(monkeypatch):
    _configure_test_pool(monkeypatch, pool_size=2)
    monkeypatch.setattr(
        speech.ta,
        "save",
        lambda buffer, audio, sample_rate, format: buffer.write(b"wav"),
    )
    long_text = ("Sentence one. Sentence two. Sentence three. " * 12).strip()

    async def scenario():
        await tts_model.initialize_model()
        lease = await tts_model.acquire_model_lease(0)

        await speech._generate_full_audio(
            lease=lease,
            text=long_text,
            voice_sample_path=Config.VOICE_SAMPLE_PATH,
            language_id=None,
            exaggeration=None,
            cfg_weight=None,
            temperature=None,
        )

        await tts_model.release_model_lease(lease)

        models = [slot.model for slot in tts_model._model_pool]
        active_models = [model for model in models if model.generated_texts]

        assert len(active_models) == 1
        assert active_models[0].name == f"model-{lease.instance_id}"
        assert len(active_models[0].generated_texts) >= 2

    asyncio.run(scenario())


def test_sse_request_keeps_one_stable_model(monkeypatch):
    _configure_test_pool(monkeypatch, pool_size=2)

    async def scenario():
        await tts_model.initialize_model()
        lease = await tts_model.acquire_model_lease(0)

        events = []
        async for event in speech.generate_speech_sse(
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

        models = [slot.model for slot in tts_model._model_pool]
        active_models = [model for model in models if model.generated_texts]

        assert len(events) >= 3
        assert len(active_models) == 1
        assert active_models[0].name == f"model-{lease.instance_id}"
        assert len(active_models[0].generated_texts) >= 2
        assert tts_model.get_pool_status()["available_instances"] == 2

    asyncio.run(scenario())
