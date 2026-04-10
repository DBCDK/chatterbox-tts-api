"""Integration tests for the active HTTP API surface."""

from tests.conftest import TEST_TEXTS


class TestHealthAndModels:
    def test_health_check(self, api_client):
        response = api_client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "config" in data
        assert "model_loaded" in data

    def test_ping(self, api_client):
        response = api_client.get("/ping")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_models_endpoint(self, api_client):
        response = api_client.get("/v1/models")
        assert response.status_code == 200

        data = response.json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        assert len(data["data"]) == 1
        assert data["data"][0]["object"] == "model"

    def test_docs_endpoints(self, api_client):
        assert api_client.get("/openapi.json").status_code == 200
        assert api_client.get("/docs").status_code == 200
        assert api_client.get("/redoc").status_code == 200


class TestSpeechEndpoint:
    def test_non_streaming_speech_returns_wav_and_usage_headers(self, api_client):
        response = api_client.post(
            "/v1/audio/speech",
            json={"input": TEST_TEXTS["short"]},
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("audio/wav")
        assert len(response.content) > 0
        assert int(response.headers["X-Usage-Input-Chars"]) == len(
            TEST_TEXTS["short"].strip()
        )
        assert float(response.headers["X-Usage-Audio-Seconds"]) > 0

    def test_tts_accepts_supported_parameters(self, api_client):
        response = api_client.post(
            "/v1/audio/speech",
            json={
                "input": TEST_TEXTS["medium"],
                "voice": "alloy",
                "response_format": "wav",
                "speed": 1.0,
                "exaggeration": 0.7,
                "cfg_weight": 0.4,
                "temperature": 0.9,
            },
        )

        assert response.status_code == 200
        assert len(response.content) > 0


class TestValidation:
    def test_missing_input_returns_422(self, api_client):
        response = api_client.post("/v1/audio/speech", json={"voice": "alloy"})
        assert response.status_code == 422

    def test_empty_input_returns_422(self, api_client):
        response = api_client.post("/v1/audio/speech", json={"input": ""})
        assert response.status_code == 422

    def test_invalid_parameter_range_returns_422(self, api_client):
        response = api_client.post(
            "/v1/audio/speech",
            json={"input": "test", "exaggeration": 5.0},
        )
        assert response.status_code == 422

    def test_text_too_long_returns_400(self, api_client):
        response = api_client.post(
            "/v1/audio/speech",
            json={"input": TEST_TEXTS["very_long"]},
        )
        assert response.status_code == 400
