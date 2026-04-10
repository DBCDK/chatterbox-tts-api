# Docker Deployment

One maintained GPU-capable Dockerfile is provided in `docker/Dockerfile`.

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

If you want to use a Hugging Face model, pass `MODEL_SOURCE=hf_repo`, `MODEL_CLASS`, and `MODEL_REPO_ID` when you run the container.
