# Docker Files

This directory contains the single maintained Dockerfile for the reduced API service.

Examples:

```bash
docker build -f docker/Dockerfile -t chatterbox-tts-api .
```

The image bakes in the two voices from `voices/`: `mic` (`voices/mic-voice.wav`,
the default) and `nic` (`voices/nic-voice.wav`). Select one per request via
the `voice` field on `POST /v1/audio/speech` (unknown names fall back to `mic`).

Run a container with different/additional voice samples mounted:

```bash
docker run --rm -p 4123:4123 \
  -v "$PWD/voices/mic-voice.wav:/app/voices/mic-voice.wav:ro" \
  -v "$PWD/voices/nic-voice.wav:/app/voices/nic-voice.wav:ro" \
  chatterbox-tts-api
```
