# Streaming API

Streaming is available through `POST /v1/audio/speech` by setting `stream_format` to `sse`.

Example request:

```json
{
  "input": "Hello from Chatterbox",
  "stream_format": "sse",
  "streaming_chunk_size": 150,
  "streaming_strategy": "sentence",
  "streaming_quality": "balanced"
}
```

## Event sequence

1. `speech.audio.info`
2. one or more `speech.audio.delta`
3. `speech.audio.done`

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

## Streaming options

- `streaming_chunk_size`: `50-500`
- `streaming_strategy`: `sentence|paragraph|fixed|word`
- `streaming_quality`: `fast|balanced|high`
