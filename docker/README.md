# Docker Files

This directory contains the single maintained Dockerfile for the reduced API service.

Examples:

```bash
docker build -f docker/Dockerfile -t chatterbox-tts-api .
```

Run a container with a mounted voice sample:

```bash
docker run --rm -p 4123:4123 \
  -v "$PWD/voice-sample.mp3:/app/voice-sample.mp3:ro" \
  chatterbox-tts-api
```
