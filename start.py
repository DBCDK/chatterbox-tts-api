#!/usr/bin/env python3
"""Development helper for the reduced API surface."""

import argparse
import subprocess
import sys


def start_dev():
    subprocess.run(
        [
            "uvicorn",
            "app.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "4123",
            "--reload",
            "--log-level",
            "debug",
        ]
    )


def start_prod():
    subprocess.run([sys.executable, "main.py"])


def run_tests():
    subprocess.run([sys.executable, "tests/run_tests.py"])


def show_info():
    print("Endpoints:")
    print("  POST /v1/audio/speech")
    print("  GET  /v1/models")
    print("  GET  /health")
    print("  GET  /ping")
    print()
    print("Useful URLs:")
    print("  http://localhost:4123/docs")
    print("  http://localhost:4123/redoc")
    print("  http://localhost:4123/health")


def main():
    parser = argparse.ArgumentParser(description="Chatterbox TTS API helper")
    parser.add_argument("command", choices=["dev", "prod", "test", "info"])
    args = parser.parse_args()

    if args.command == "dev":
        start_dev()
    elif args.command == "prod":
        start_prod()
    elif args.command == "test":
        run_tests()
    elif args.command == "info":
        show_info()


if __name__ == "__main__":
    main()
