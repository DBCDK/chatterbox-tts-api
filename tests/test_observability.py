"""Unit tests for structured logging utilities."""

import asyncio
import io
import json
import logging

import pytest

from app.api.endpoints import speech
from app.core.observability import JsonLogFormatter, log_event


@pytest.fixture(scope="session", autouse=True)
def check_api_health():
    return None


def _make_logger(name: str) -> tuple[logging.Logger, io.StringIO]:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(
        JsonLogFormatter(service="chatterbox-tts-api", version="test-version")
    )
    logger = logging.getLogger(name)
    logger.handlers = []
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger, stream


def test_json_log_formatter_outputs_structured_payload():
    logger, stream = _make_logger("tests.observability.formatter")

    log_event(
        logger,
        logging.INFO,
        "request_completed",
        request_id="req-123",
        request_mode="audio",
        route="/v1/audio/speech",
        outcome="success",
        input_chars=42,
    )

    payload = json.loads(stream.getvalue().strip())
    assert payload["event"] == "request_completed"
    assert payload["message"] == "request_completed"
    assert payload["service"] == "chatterbox-tts-api"
    assert payload["version"] == "test-version"
    assert payload["request_id"] == "req-123"
    assert payload["route"] == "/v1/audio/speech"
    assert payload["input_chars"] == 42


def test_request_log_helper_emits_expected_fields(monkeypatch):
    logger, stream = _make_logger("tests.observability.request")
    monkeypatch.setattr(speech, "logger", logger)

    async def scenario():
        context = speech._new_request_context(mode="sse")
        speech._log_request_event(
            logging.INFO,
            "request_started",
            context,
            outcome="started",
            input_chars=128,
        )

    asyncio.run(scenario())

    payload = json.loads(stream.getvalue().strip())
    assert payload["event"] == "request_started"
    assert payload["request_mode"] == "sse"
    assert payload["route"] == "/v1/audio/speech"
    assert payload["outcome"] == "started"
    assert payload["input_chars"] == 128
    assert "request_id" in payload
    assert "elapsed_seconds" in payload


def test_request_logs_do_not_include_raw_input_text(monkeypatch):
    logger, stream = _make_logger("tests.observability.no_text")
    monkeypatch.setattr(speech, "logger", logger)
    sample_text = "Do not leak this request text"

    async def scenario():
        context = speech._new_request_context(mode="audio")
        speech._log_request_event(
            logging.INFO,
            "request_started",
            context,
            outcome="started",
            input_chars=len(sample_text),
        )

    asyncio.run(scenario())

    output = stream.getvalue()
    assert sample_text not in output
    payload = json.loads(output.strip())
    assert payload["input_chars"] == len(sample_text)
    assert "input" not in payload
