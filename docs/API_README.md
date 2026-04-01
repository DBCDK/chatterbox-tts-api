# API Reference

## Endpoints

- `POST /v1/audio/speech`
- `GET /v1/models`
- `GET /health`
- `GET /ping`

## `POST /v1/audio/speech`

Request body:

```json
{
  "input": "Text to convert to speech",
  "voice": "alloy",
  "response_format": "wav",
  "speed": 1.0,
  "stream_format": "audio",
  "exaggeration": 0.7,
  "cfg_weight": 0.4,
  "temperature": 0.9,
  "streaming_chunk_size": 150,
  "streaming_strategy": "sentence",
  "streaming_quality": "balanced"
}
```

Notes:

- `voice` is accepted for compatibility and resolves to the configured sample path
- `response_format` is accepted for compatibility; the service currently returns WAV
- `speed` is accepted for compatibility and ignored
- `stream_format` accepts `audio` or `sse`

### Non-streaming response

- content type: `audio/wav`
- response headers:
  - `X-Usage-Input-Chars`
  - `X-Usage-Audio-Seconds`

### SSE response

- content type: `text/event-stream`
- event types:
  - `speech.audio.info`
  - `speech.audio.delta`
  - `speech.audio.done`

Final event shape:

```json
{
  "type": "speech.audio.done",
  "usage": {
    "input_chars": 123,
    "audio_seconds": 4.56
  }
}
```

## `GET /v1/models`

OpenAI-style model listing for the currently configured Chatterbox model.

## `GET /health`

Returns startup and model-loading state plus a small configuration summary.
