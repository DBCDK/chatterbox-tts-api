"""Streaming integration tests for the reduced speech endpoint."""

import json

from conftest import TEST_TEXTS


class TestSpeechStreaming:
    def test_sse_streaming_returns_expected_events(self, api_client):
        response = api_client.post(
            "/v1/audio/speech",
            json={
                "input": TEST_TEXTS["medium"],
                "stream_format": "sse",
                "streaming_strategy": "sentence",
                "streaming_chunk_size": 150,
            },
            stream=True,
            headers={"Accept": "text/event-stream"},
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        event_types = []
        done_event = None

        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue

            event = json.loads(line[6:])
            event_types.append(event["type"])

            if event["type"] == "speech.audio.done":
                done_event = event
                break

        assert "speech.audio.info" in event_types
        assert "speech.audio.delta" in event_types
        assert done_event is not None
        assert done_event["usage"]["input_chars"] == len(TEST_TEXTS["medium"].strip())
        assert done_event["usage"]["audio_seconds"] > 0

    def test_invalid_stream_format_returns_422(self, api_client):
        response = api_client.post(
            "/v1/audio/speech",
            json={"input": TEST_TEXTS["short"], "stream_format": "invalid"},
        )
        assert response.status_code == 422
