#!/usr/bin/env python3
"""Minimal test runner for the reduced API surface."""

import argparse
import os
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Chatterbox TTS API integration tests"
    )
    parser.add_argument("--api-url", default="http://localhost:4123")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--streaming-only", action="store_true")
    args = parser.parse_args()

    os.environ["CHATTERBOX_TEST_URL"] = args.api_url
    os.environ["TEST_TIMEOUT"] = str(args.timeout)

    cmd = [sys.executable, "-m", "pytest"]
    if args.streaming_only:
        cmd.append("tests/test_streaming.py")
    else:
        cmd.extend(["tests/test_api.py", "tests/test_streaming.py"])
    if args.verbose:
        cmd.append("-v")

    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
