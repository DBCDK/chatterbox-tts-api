"""Shared pytest fixtures for the reduced API surface."""

import os
import time

import pytest
import requests


BASE_URL = os.getenv("CHATTERBOX_TEST_URL", "http://localhost:4123")
TEST_TIMEOUT = int(os.getenv("TEST_TIMEOUT", "120"))
HEALTH_TIMEOUT = int(os.getenv("API_HEALTH_TIMEOUT", "5"))

TEST_TEXTS = {
    "short": "Hello, this is a simple test.",
    "medium": "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.",
    "very_long": "This is a test. " * 400,
}


class APIClient:
    def __init__(self, base_url: str = BASE_URL, timeout: int = TEST_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get(self, endpoint: str, **kwargs) -> requests.Response:
        timeout = kwargs.pop("timeout", self.timeout)
        return requests.get(
            f"{self.base_url}/{endpoint.lstrip('/')}", timeout=timeout, **kwargs
        )

    def post(self, endpoint: str, **kwargs) -> requests.Response:
        timeout = kwargs.pop("timeout", self.timeout)
        return requests.post(
            f"{self.base_url}/{endpoint.lstrip('/')}", timeout=timeout, **kwargs
        )

    def is_healthy(self) -> bool:
        try:
            response = self.get("/health", timeout=HEALTH_TIMEOUT)
            return response.status_code == 200
        except Exception:
            return False

    def wait_for_health(self, max_attempts: int = 10, delay: float = 1.0) -> bool:
        for attempt in range(max_attempts):
            if self.is_healthy():
                return True
            if attempt < max_attempts - 1:
                time.sleep(delay)
        return False


@pytest.fixture(scope="session")
def api_client() -> APIClient:
    return APIClient()


@pytest.fixture(scope="session", autouse=True)
def check_api_health(api_client: APIClient):
    if not api_client.wait_for_health():
        pytest.skip(f"API not available at {BASE_URL}. Please start the server first.")


def pytest_configure(config):
    config.addinivalue_line("markers", "api: API integration tests")
    config.addinivalue_line("markers", "streaming: streaming integration tests")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "test_api" in item.nodeid:
            item.add_marker(pytest.mark.api)
        if "test_streaming" in item.nodeid:
            item.add_marker(pytest.mark.streaming)
            item.add_marker(pytest.mark.api)
