# Docker Files

This directory contains build-only Dockerfiles for the reduced API service.

Examples:

```bash
docker build -f docker/Dockerfile -t chatterbox-tts-api .
docker build -f docker/Dockerfile.uv -t chatterbox-tts-api-uv .
docker build -f docker/Dockerfile.gpu -t chatterbox-tts-api-gpu .
```

Run a container with a mounted voice sample:

```bash
docker run --rm -p 4123:4123 \
  -v "$PWD/voice-sample.mp3:/app/voice-sample.mp3:ro" \
  chatterbox-tts-api
```
