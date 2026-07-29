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
  "voice": "mic",
  "response_format": "wav",
  "speed": 1.0,
  "stream_format": "audio",
  "exaggeration": 0.7,
  "cfg_weight": 0.4,
  "temperature": 0.9
}
```

Notes:

- `voice` resolves to the configured sample path; unknown names fall back to the default voice
- `response_format` accepts `pcm` or `wav` (default `wav`). Other OpenAI-documented values
  (`mp3`, `opus`, `aac`, `flac`) are rejected with a 400 -- this service does not run a lossy
  encoder in the request path
- `speed` is accepted for compatibility and ignored
- `stream_format` accepts `audio` (a genuine chunked byte stream) or `sse` (debug-only, see
  [STREAMING_API.md](STREAMING_API.md)); absent (the default) returns one buffered response

### Wire format -- all three responses

Every response, streamed or buffered, carries the same 16-bit signed little-endian integer
PCM audio, described by headers present on all three:

- `X-Audio-Sample-Rate`
- `X-Audio-Channels`
- `X-Audio-Bits-Per-Sample`

### Non-streaming response (no `stream_format`)

- content type: `audio/wav` or `audio/pcm`, depending on `response_format`
- response headers (in addition to the wire-format headers above):
  - `X-Usage-Input-Chars`
  - `X-Usage-Audio-Seconds`

### Chunked audio response (`stream_format: "audio"`)

- content type: `audio/wav` or `audio/pcm`, depending on `response_format`
- a `wav` response opens with a streaming RIFF header (unknown-length placeholder sizes,
  since the total duration isn't known until generation finishes); a `pcm` response has no
  header at all -- just raw samples
- `X-Usage-Audio-Seconds` is not sent on this exit: duration isn't known when headers go out.
  Derive usage from byte count instead: `bytes / (sample_rate * channels * 2)`

### SSE response (`stream_format: "sse"`)

- content type: `text/event-stream`
- debug-only: nothing in production consumes this path
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
