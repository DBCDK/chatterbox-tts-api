# Streaming API

`POST /v1/audio/speech` has two streaming forms, selected by `stream_format`, plus the default
buffered response when it's absent. All three carry 16-bit signed little-endian integer PCM;
see [API_README.md](API_README.md) for the shared wire-format headers.

## Chunked audio (`stream_format: "audio"`)

The production streaming path: a genuine chunked HTTP body, one chunk per generated sentence,
no base64 or JSON framing overhead. This is what a real client should use.

```json
{
  "input": "Hello from Chatterbox",
  "stream_format": "audio",
  "response_format": "pcm"
}
```

- `response_format: "pcm"` -- headerless raw samples, nothing to parse before the first chunk
- `response_format: "wav"` (default) -- a streaming RIFF/WAV header with unknown-length
  placeholder sizes (`0xFFFFFFFF`), since the total duration isn't known until generation
  finishes. Accepted by ffmpeg, ffplay, and browsers; some strict parsers reject unknown-length
  WAV, so prefer `pcm` if a downstream consumer turns out to be one of those
- No `X-Usage-Audio-Seconds` header -- duration isn't known when headers are sent. Derive usage
  from the response byte count instead: `bytes / (sample_rate * channels * 2)`. This is exactly
  how glyph-gate meters usage, and it's why the 16-bit-PCM contract is load-bearing: at 32-bit
  float the same byte count would read as half the true duration

## SSE (`stream_format: "sse"`)

Debug-only. Nothing in production consumes this path -- prefer chunked `audio` above for any
real client.

```json
{
  "input": "Hello from Chatterbox",
  "stream_format": "sse"
}
```

### Event sequence

1. `speech.audio.info`
2. one or more `speech.audio.delta`
3. `speech.audio.done`

`speech.audio.info` is self-describing (added so the delta encoding doesn't need to be inferred):

```json
{
  "type": "speech.audio.info",
  "sample_rate": 24000,
  "channels": 1,
  "bits_per_sample": 16,
  "format": "pcm"
}
```

Each `speech.audio.delta` carries base64-encoded 16-bit PCM for one sentence:

```json
{
  "type": "speech.audio.delta",
  "audio": "<base64>"
}
```

Example completion event:

```json
{
  "type": "speech.audio.done",
  "usage": {
    "input_chars": 123,
    "audio_seconds": 4.56
  }
}
```
