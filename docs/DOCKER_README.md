# Docker Deployment

Only Dockerfiles are maintained in `docker/`.

## Quick start

```bash
docker build -f docker/Dockerfile -t chatterbox-tts-api .
docker run --rm -p 4123:4123 \
  -v "$PWD/voice-sample.mp3:/app/voice-sample.mp3:ro" \
  chatterbox-tts-api
```

Test the API:

```bash
curl -X POST http://localhost:4123/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello from Docker"}' \
  --output test.wav
```

Other image variants can be built from the other Dockerfiles in `docker/`, such as `Dockerfile.uv`, `Dockerfile.gpu`, `Dockerfile.cpu`, and `Dockerfile.blackwell`.
