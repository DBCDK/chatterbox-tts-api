# Chatterbox TTS API

Minimal FastAPI wrapper around Chatterbox TTS with:

- `POST /v1/audio/speech`
- `GET /v1/models`
- `GET /health`
- startup model loading
- default, Hugging Face repo, and local model loading
- non-streaming WAV responses and SSE streaming
- per-response usage metadata: `input_chars` and audio duration

## Quick Start

```bash
git clone https://github.com/travisvn/chatterbox-tts-api
cd chatterbox-tts-api
cp .env.example .env
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 4123
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 4123
```

## API

### Non-streaming

```bash
curl -X POST http://localhost:4123/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello from Chatterbox"}' \
  --output speech.wav -i
```

Response headers include:

- `X-Usage-Input-Chars`
- `X-Usage-Audio-Seconds`

### SSE streaming

```bash
curl -X POST http://localhost:4123/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"input":"Hello from Chatterbox","stream_format":"sse"}'
```

The final `speech.audio.done` event includes:

- `usage.input_chars`
- `usage.audio_seconds`

### Models

```bash
curl http://localhost:4123/v1/models
```

### Health

```bash
curl http://localhost:4123/health
```

## Configuration

Important settings:

- `VOICE_SAMPLE_PATH`
- `DEVICE`
- `MODEL_SOURCE`
- `MODEL_CLASS`
- `MODEL_REPO_ID`
- `MODEL_LOCAL_PATH`
- `MODEL_SUPPORTED_LANGUAGES`
- `DEFAULT_LANGUAGE`
- `EXAGGERATION`
- `CFG_WEIGHT`
- `TEMPERATURE`
- `MAX_CHUNK_LENGTH`
- `MAX_TOTAL_LENGTH`

See `.env.example` and `.env.example.docker` for full examples.

## Model Sources

### Default bundled loader

```env
MODEL_SOURCE=default
USE_MULTILINGUAL_MODEL=true
```

### Hugging Face repo

```env
MODEL_SOURCE=hf_repo
MODEL_CLASS=multilingual
MODEL_REPO_ID=CoRal-project/roest-v3-chatterbox-500m
MODEL_SUPPORTED_LANGUAGES=da,en
DEFAULT_LANGUAGE=da
```

### Local snapshot directory

```env
MODEL_SOURCE=local_dir
MODEL_CLASS=multilingual
MODEL_LOCAL_PATH=./models/coral-roest-v3
MODEL_SUPPORTED_LANGUAGES=da,en
DEFAULT_LANGUAGE=da
```

## Docker

Only Dockerfiles are maintained in `docker/`.

Build the variant you want directly, for example:

```bash
docker build -f docker/Dockerfile -t chatterbox-tts-api .
docker run --rm -p 4123:4123 \
  -v "$PWD/voice-sample.mp3:/app/voice-sample.mp3:ro" \
  chatterbox-tts-api
```

## Docs

- `docs/API_README.md`
- `docs/STREAMING_API.md`
- `docs/MULTILINGUAL.md`
- `docs/DOCKER_README.md`

## Tests

```bash
python tests/run_tests.py
```
