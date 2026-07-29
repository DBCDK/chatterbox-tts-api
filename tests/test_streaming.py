"""Streaming integration tests for the reduced speech endpoint."""

import json
import shutil
import struct
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tests.conftest import TEST_TEXTS

# Real production text from the Phase 0 corpus (11 sentences, ~1200 chars): long enough
# to exercise multi-chunk streaming and to keep a lease busy for the overload test,
# short enough to stay under MAX_TOTAL_LENGTH (unlike TEST_TEXTS["very_long"], which is
# deliberately oversized to trigger the 400 tested in test_api.py).
_CORPUS_PATH = Path(__file__).parent / "audio_quality" / "corpus.json"
_CORPUS = json.loads(_CORPUS_PATH.read_text(encoding="utf-8"))
MULTI_SENTENCE_TEXT = next(e for e in _CORPUS if e["id"] == "lokalplan-01")["text"]

FFPROBE_AVAILABLE = shutil.which("ffprobe") is not None


def _metric_value(metrics_text: str, metric_name: str, **labels) -> float:
    """Read a single Prometheus sample's value, matched by metric name and labels.

    Doesn't try to be a general Prometheus text parser -- just enough to pull one
    `_sum`/counter value out of a real /metrics response for a before/after delta.
    """
    label_fragments = [f'{key}="{value}"' for key, value in labels.items()]
    for line in metrics_text.splitlines():
        if not line.startswith(metric_name + "{"):
            continue
        if all(fragment in line for fragment in label_fragments):
            return float(line.rsplit(" ", 1)[1])
    raise AssertionError(f"no {metric_name} sample found for labels {labels}")


def _probe_audio(path: Path, extra_args: tuple[str, ...] = ()) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            *extra_args,
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"ffprobe failed: {result.stderr}"
    payload = json.loads(result.stdout)
    assert payload["streams"], f"ffprobe found no audio streams: {result.stderr}"
    return payload["streams"][0]


class TestSpeechStreaming:
    def test_sse_streaming_returns_expected_events(self, api_client):
        response = api_client.post(
            "/v1/audio/speech",
            json={
                "input": TEST_TEXTS["medium"],
                "stream_format": "sse",
            },
            stream=True,
            headers={"Accept": "text/event-stream"},
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        event_types = []
        info_event = None
        done_event = None

        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue

            event = json.loads(line[6:])
            event_types.append(event["type"])

            if event["type"] == "speech.audio.info":
                info_event = event
            if event["type"] == "speech.audio.done":
                done_event = event
                break

        assert "speech.audio.info" in event_types
        assert "speech.audio.delta" in event_types
        assert info_event is not None
        assert info_event["format"] == "pcm"
        assert info_event["bits_per_sample"] == 16
        assert done_event is not None
        assert done_event["usage"]["input_chars"] == len(TEST_TEXTS["medium"].strip())
        assert done_event["usage"]["audio_seconds"] > 0

    def test_invalid_stream_format_returns_422(self, api_client):
        response = api_client.post(
            "/v1/audio/speech",
            json={"input": TEST_TEXTS["short"], "stream_format": "invalid"},
        )
        assert response.status_code == 422

    def test_unsupported_response_format_returns_400(self, api_client):
        response = api_client.post(
            "/v1/audio/speech",
            json={"input": TEST_TEXTS["short"], "response_format": "mp3"},
        )
        assert response.status_code == 400
        body = response.json()
        assert "mp3" in body["error"]["message"]


class TestExitAWavContract:
    """Exit A (buffered, no stream_format): the same WAV byte layout the unit tests
    assert in isolation, verified here against a real HTTP response."""

    def test_wav_header_and_body_are_self_consistent_16bit_pcm(self, api_client):
        response = api_client.post(
            "/v1/audio/speech",
            json={"input": TEST_TEXTS["short"], "response_format": "wav"},
        )
        assert response.status_code == 200
        content = response.content

        (
            riff, chunk_size, wave, fmt_tag, subchunk_size, audio_format, channels,
            sample_rate, byte_rate, block_align, bits_per_sample, data_tag, data_size,
        ) = struct.unpack("<4sI4s4sIHHIIHH4sI", content[:44])

        assert riff == b"RIFF"
        assert wave == b"WAVE"
        assert fmt_tag == b"fmt "
        assert subchunk_size == 16
        assert audio_format == 1  # integer PCM, not float
        assert channels == 1
        assert bits_per_sample == 16
        assert byte_rate == sample_rate * 2
        assert block_align == 2
        assert data_tag == b"data"
        # This is the "half the bytes at identical duration" acceptance criterion,
        # checked via internal self-consistency rather than a second live generation:
        # model sampling means two independent calls for the same text can differ
        # slightly in length, so comparing this response's own declared sizes against
        # its own body avoids a flaky cross-request byte comparison.
        assert chunk_size == 36 + data_size
        assert data_size == len(content) - 44
        assert int(response.headers["X-Audio-Bits-Per-Sample"]) == 16


class TestExitBStreaming:
    """Exit B (stream_format=audio): genuine chunked bytes, verified against real
    network delivery timing and the server's own usage metrics."""

    def test_delivers_first_bytes_before_generation_completes(self, api_client):
        start = time.perf_counter()
        response = api_client.post(
            "/v1/audio/speech",
            json={
                "input": MULTI_SENTENCE_TEXT,
                "stream_format": "audio",
                "response_format": "pcm",
            },
            stream=True,
        )
        assert response.status_code == 200

        first_chunk_at = None
        for chunk in response.iter_content(chunk_size=None):
            if chunk and first_chunk_at is None:
                first_chunk_at = time.perf_counter()
        total_elapsed = time.perf_counter() - start

        assert first_chunk_at is not None, "stream produced no bytes"
        ttfb = first_chunk_at - start
        # The multi-sentence corpus text takes several chunks to generate; a buffered
        # response would put ttfb ~= total_elapsed, so this margin is what actually
        # distinguishes genuine streaming from Exit A's behavior.
        assert ttfb < total_elapsed * 0.7, (
            f"first byte at {ttfb:.3f}s of {total_elapsed:.3f}s total -- "
            "looks buffered, not streamed"
        )

    def test_byte_derived_audio_seconds_matches_server_metric(self, api_client):
        before = api_client.get("/metrics").text

        response = api_client.post(
            "/v1/audio/speech",
            json={
                "input": MULTI_SENTENCE_TEXT,
                "stream_format": "audio",
                "response_format": "pcm",
            },
            stream=True,
        )
        assert response.status_code == 200
        sample_rate = int(response.headers["X-Audio-Sample-Rate"])
        body = response.content

        after = api_client.get("/metrics").text

        byte_derived_seconds = len(body) / (sample_rate * 1 * 2)
        server_delta_seconds = _metric_value(
            after, "chatterbox_tts_audio_seconds_sum", mode="audio_stream"
        ) - _metric_value(before, "chatterbox_tts_audio_seconds_sum", mode="audio_stream")

        # "Within one frame" per the Phase 1 plan's metering contract.
        assert abs(byte_derived_seconds - server_delta_seconds) < 1.0 / sample_rate


class TestFfprobeParseability:
    """All three exits must produce audio a real downstream tool can parse -- this is
    what the unit-level header-byte assertions can't prove by themselves."""

    def test_exit_a_buffered_wav(self, api_client, tmp_path):
        if not FFPROBE_AVAILABLE:
            pytest.skip("ffprobe not installed")

        response = api_client.post(
            "/v1/audio/speech",
            json={"input": TEST_TEXTS["short"], "response_format": "wav"},
        )
        assert response.status_code == 200
        wav_path = tmp_path / "exit_a.wav"
        wav_path.write_bytes(response.content)

        stream = _probe_audio(wav_path)
        assert stream["codec_name"] == "pcm_s16le"
        assert stream["channels"] == 1
        assert str(stream["sample_rate"]) == response.headers["X-Audio-Sample-Rate"]

    def test_exit_b_streamed_wav(self, api_client, tmp_path):
        if not FFPROBE_AVAILABLE:
            pytest.skip("ffprobe not installed")

        response = api_client.post(
            "/v1/audio/speech",
            json={
                "input": TEST_TEXTS["short"],
                "stream_format": "audio",
                "response_format": "wav",
            },
            stream=True,
        )
        assert response.status_code == 200
        wav_path = tmp_path / "exit_b.wav"
        wav_path.write_bytes(response.content)

        # Unknown-length RIFF header (0xFFFFFFFF sizes) -- confirms ffmpeg/ffprobe
        # accepts it, per the plan's note that support is widespread but not universal.
        stream = _probe_audio(wav_path)
        assert stream["codec_name"] == "pcm_s16le"
        assert stream["channels"] == 1

    def test_exit_b_streamed_pcm(self, api_client, tmp_path):
        if not FFPROBE_AVAILABLE:
            pytest.skip("ffprobe not installed")

        response = api_client.post(
            "/v1/audio/speech",
            json={
                "input": TEST_TEXTS["short"],
                "stream_format": "audio",
                "response_format": "pcm",
            },
            stream=True,
        )
        assert response.status_code == 200
        sample_rate = response.headers["X-Audio-Sample-Rate"]
        pcm_path = tmp_path / "exit_b.pcm"
        pcm_path.write_bytes(response.content)

        # Raw PCM has no self-describing header -- ffprobe needs explicit format hints.
        stream = _probe_audio(
            pcm_path, extra_args=["-f", "s16le", "-ar", sample_rate, "-ac", "1"]
        )
        assert stream["codec_name"] == "pcm_s16le"


class TestDisconnectAndOverload:
    def test_client_disconnect_releases_lease(self, api_client):
        baseline = api_client.get("/health").json()["pool_status"]["available_instances"]

        response = api_client.post(
            "/v1/audio/speech",
            json={
                "input": MULTI_SENTENCE_TEXT,
                "stream_format": "audio",
                "response_format": "pcm",
            },
            stream=True,
        )
        assert response.status_code == 200
        chunks_read = 0
        for _ in response.iter_content(chunk_size=None):
            chunks_read += 1
            if chunks_read >= 2:
                break
        response.close()

        for _ in range(20):
            pool_status = api_client.get("/health").json()["pool_status"]
            if pool_status["available_instances"] == baseline:
                return
            time.sleep(0.5)
        pytest.fail("lease was not released within 10s of client disconnect")

    def test_overload_returns_503_before_any_bytes_sent(self, api_client):
        health = api_client.get("/health").json()
        pool_size = health["config"]["model_instance_count"]
        # _acquire_request_lease waits up to min(MAX_QUEUE_WAIT_SECONDS, remaining
        # deadline) for a lease before rejecting -- on a real deployment that can
        # legitimately exceed the client's default 120s timeout under GPU contention,
        # which reads as a bare connection timeout, not a 503. Size the client's
        # per-request timeout off the server's own config instead of guessing.
        client_timeout = (
            health["config"]["request_timeout_seconds"]
            + health["config"]["max_queue_wait_seconds"]
            + 30
        )

        def make_request(_):
            return api_client.post(
                "/v1/audio/speech",
                json={"input": MULTI_SENTENCE_TEXT},
                timeout=client_timeout,
            )

        # One request per pool slot to occupy every instance, plus one that must
        # overflow -- MAX_QUEUE_WAIT_SECONDS on this deployment determines whether it
        # rejects immediately or after a short queue wait.
        with ThreadPoolExecutor(max_workers=pool_size + 1) as pool:
            responses = list(pool.map(make_request, range(pool_size + 1)))

        statuses = [r.status_code for r in responses]
        assert 503 in statuses, f"expected an overload rejection, got {statuses}"
        overloaded = next(r for r in responses if r.status_code == 503)
        assert overloaded.headers["content-type"].startswith("application/json")
        body = overloaded.json()
        assert "error" in body
