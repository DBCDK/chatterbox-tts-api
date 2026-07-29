# Phase 1 — Wire Protocol And Framing Seam

Part of the streaming redesign. See `prompts/streaming-redesign-plan.md` for the overall plan and the
resolved design decisions. This document covers Phase 1 only.

Depends on Phase 0 (`prompts/streaming-phase-0-plan.md`) for the reference corpus and the fixed test
harness.

## Context

`POST /v1/audio/speech` currently has two paths that diverge at the model call:

- **`stream_format` absent or `"audio"`** — calls `model.generate(...)`, waits for the entire text,
  writes a complete WAV via `torchaudio.save`, then copies the buffer again before sending. Despite
  the name, `"audio"` does not stream.
- **`stream_format="sse"`** — calls `model.generate_stream_async(...)`, base64-encodes each yielded
  tensor as int16 PCM, and sends it as a `speech.audio.delta` event.

Consequences of that split:

- The two paths return **different sample formats** for the same audio: 32-bit float WAV
  (`audioFormat: 3`, 96,000 B/s) versus 16-bit int PCM (48,000 B/s).
- Timeout handling, disconnect handling, metrics, and logging are implemented twice, which is most of
  why `app/api/endpoints/speech.py` is 766 lines.
- `response_format` is accepted and silently ignored — a client asking for `mp3` receives WAV.
- There is no genuine byte-streaming transport at all.

## Objective

Collapse both paths onto **one production pipeline with three exits**, standardize on 16-bit PCM, and
add a real chunked-bytes transport.

This phase touches **no model code**. The audio source remains sentence-granular
`generate_stream_async`; Phase 3 changes that behind the same seam.

## Non-Goals

- Compressed formats. `opus`, `mp3`, `aac`, `flac` are out — both services run in the same k8s cluster
  so bandwidth is not a constraint, and an encoder subprocess is risk without reward.
- Token-level streaming. Phase 3.
- Changing the model pool or lease semantics.
- Perfecting the SSE event schema. Per the resolved compatibility decision, SSE becomes a debug-only
  path once glyph-gate streams over `audio`.

## The Seam

```
AudioSource      async iterator of float32 tensors, plus sample_rate
    |            (Phase 1: sentence-granular. Phase 3: sub-sentence. Same interface.)
    v
Framer           float32 tensor in, bytes out. Streaming, with finalize().
    |            (PcmFramer | WavStreamFramer. An OpusEncoder would slot in here.)
    v
Exit             buffered WAV | chunked HTTP body | base64 SSE deltas
```

Sketch:

```python
class Framer(Protocol):
    media_type: str
    def header(self) -> bytes: ...              # b"" for raw PCM
    def frame(self, chunk: torch.Tensor) -> bytes: ...
    def finalize(self) -> bytes: ...            # b"" for both current framers
```

The point of the seam is that Phase 3 swaps only the source, and a future codec adds only a framer.
Neither should require touching transport or error handling.

## Work Item 1 — Request Model

In `app/models/requests.py`:

- Add `response_format: Literal["pcm", "wav"] = "wav"`.
- **Reject** `mp3`, `opus`, `aac`, `flac` with a 400 carrying a clear message. Today these silently
  return WAV; failing loudly is the point. Note this in `CHANGELOG.md` as a behaviour change.
- Keep `stream_format` as `audio | sse`, absent meaning non-streaming.
- Keep accepting and ignoring `speed`, but say so in the field description rather than implying
  support.
- The existing validators use Pydantic v1 style (`@validator`). Match the surrounding code rather
  than mixing styles; migrating the file to v2 validators is a separate cleanup.

`wav` remains the default even though OpenAI defaults to `mp3`. That divergence is deliberate and
documented — adopting `mp3` would put a lossy encoder in the path of every request.

## Work Item 2 — Framing Module

New file `app/core/audio/framing.py`.

### Sample Format — Non-Negotiable

**16-bit signed little-endian integer PCM on every exit.**

The model emits float32 and `torchaudio.save` preserves it, which is how the non-streaming path ended
up emitting 32-bit float. Conversion is the same as the current SSE path:

```python
pcm = (torch.clamp(chunk, -1.0, 1.0) * 32767).to(torch.int16)
```

Two hazards, both worth a test:

1. **Do not use `torchaudio.save` for the WAV header.** Given a float32 tensor it writes
   `audioFormat: 3` at 32 bits. Write the header by hand.
2. **Endianness must be explicit.** The current SSE path relies on native-endian `.tobytes()`, which
   is an unstated contract. Little-endian is the WAV standard and what every consumer expects — assert
   it rather than inherit it from the host architecture.

The second hazard has a billing consequence. glyph-gate derives usage as
`payload_bytes / (sample_rate * channels * bytes_per_sample)`. At 32-bit float that reads **half** the
true duration. A 2x metering error arising from a library default is exactly the kind of bug that
ships unnoticed, so the header assertion test is not optional.

### `PcmFramer`

- `media_type = "audio/pcm"` — not IANA-registered, but what OpenAI and compatible servers use.
  Avoid `audio/L16`: it *is* registered and specifies **big**-endian, contradicting our contract.
- `header()` returns `b""`.
- `frame()` returns int16 LE bytes.

### `WavStreamFramer`

44-byte canonical header, written by hand:

| Offset | Bytes | Value |
|---|---|---|
| 0 | 4 | `RIFF` |
| 4 | 4 | chunk size — `0xFFFFFFFF` when streaming, `36 + data_len` when buffered |
| 8 | 4 | `WAVE` |
| 12 | 4 | `fmt ` (trailing space) |
| 16 | 4 | `16` — subchunk size |
| 20 | 2 | `1` — audioFormat, integer PCM |
| 22 | 2 | `1` — channels |
| 24 | 4 | `24000` — sample rate |
| 28 | 4 | `48000` — byte rate, `sr * ch * bits/8` |
| 32 | 2 | `2` — block align, `ch * bits/8` |
| 34 | 2 | `16` — bits per sample |
| 36 | 4 | `data` |
| 40 | 4 | data size — `0xFFFFFFFF` when streaming, actual when buffered |

Use `0xFFFFFFFF` for unknown lengths, matching what ffmpeg's wav muxer writes to non-seekable output.
Some strict parsers reject it; ffmpeg, ffplay, and browsers accept it. The buffered exit knows the true
length and must write real values.

Take sample rate from `lease.model.sr` rather than hardcoding 24000 — it is a model property.

## Work Item 3 — One Pipeline, Three Exits

`app/api/endpoints/speech.py` is 766 lines and carries near-duplicate error handling per path. Split
it; a new `app/api/endpoints/speech_stream.py` for transports, with shared setup staying put, is a
reasonable shape.

### Shared, Unchanged

Keep in this order, exactly as today:

1. Resolve voice and language (`resolve_voice_path_and_language`).
2. Validate language and text length.
3. Build the request context (`_new_request_context`) with its 120 s deadline.
4. **Acquire the lease before the response begins** (`speech.py:633`). This is why overload returns a
   clean 503 instead of a truncated stream. Do not move it into the generator.
5. Preserve the deadline and disconnect guards (`_guard_request_state`) between chunks.

### Exit A — Buffered, No `stream_format`

Drain the streaming source, concatenate, and send one complete WAV with real header sizes and exact
`X-Usage-Audio-Seconds`.

This replaces the separate `generate()` call, which is what collapses the two paths. Coral's own
docstring states the equivalence:

> Concatenating all yielded tensors produces the same result as `generate()`.

That is a docstring, not a guarantee. **Add a test asserting it** — generate the same seeded input both
ways and compare. If it fails, the buffered exit must keep calling `generate()` and the paths stay
split; better to discover that in a test than in production audio.

Removing the double copy at `speech.py:682` falls out of this rewrite.

### Exit B — Chunked Bytes, `stream_format="audio"`

Send `header()`, then `frame()` per chunk, then `finalize()`. Media type from the framer.

### Exit C — SSE, `stream_format="sse"`

Keep the current event sequence. Add a `format` field to `speech.audio.info` so the delta encoding is
self-describing. Deltas carry base64 of the **framed** bytes.

Do not invest further here — nothing in production will use it.

## Work Item 4 — Response Headers

Exit B cannot report `X-Usage-Audio-Seconds`, since duration is unknown when headers are sent. Instead
emit the format parameters, all of which are known before generation starts:

```
X-Audio-Sample-Rate: 24000
X-Audio-Channels: 1
X-Audio-Bits-Per-Sample: 16
```

glyph-gate counts bytes and divides. No HTTP trailers — support across k8s ingress and service meshes
is inconsistent, and a silently dropped trailer means silently lost metering.

Keep `X-Usage-Input-Chars`, `X-Model-Instance-ID`, and `X-Request-ID` on all exits. Keep
`Cache-Control: no-cache` and `X-Accel-Buffering: no` on the streaming exits.

Emit the `X-Audio-*` headers on **all three exits**, not just B. They cost nothing, and a client can
then use one code path regardless of mode.

## Work Item 5 — Deletions

- `generate_speech_internal` (`speech.py:389`) — dead code. Exported but called nowhere; its consumer
  (the long-text backend) was already removed per `prompts/cleanup-plan.md`. Drop it from `__all__`
  too.
- The `io.BytesIO(buffer.getvalue())` copy at `speech.py:682`, subsumed by Exit A.

## Tests

Unit — must pass with no server running, which requires the Phase 0 conftest fix:

- WAV header byte-for-byte: `audioFormat == 1`, `bitsPerSample == 16`, `byteRate == 48000`,
  `blockAlign == 2`. Both the streaming (`0xFFFFFFFF`) and buffered (real size) variants.
- Little-endian output verified against a known sample value, not against `.tobytes()` on the host.
- `PcmFramer.header()` is empty; framed byte count is exactly `frames * 2`.
- Clamping: samples at ±1.5 do not wrap around to the opposite sign.
- Framer behaviour on an empty source (no sentences) — header still valid, or a clean error.

Integration — against a running server:

- All three exits produce audio that `ffprobe` parses, with the expected format on each.
- Exit B delivers first bytes before generation completes. Assert on time-to-first-byte, not just
  eventual success.
- `audio_seconds` derived from Exit B's byte count matches the server-side
  `chatterbox_tts_audio_seconds` value to within one frame, across the Phase 0 corpus. This is the
  metering contract — test it explicitly, not incidentally.
- Exit A output is byte-identical in duration to the Phase 0 baseline, at half the byte count.
- `response_format: "mp3"` returns 400 with a useful message.
- Client disconnect mid-stream releases the lease and records a `disconnect` outcome. Existing
  `ClientDisconnected` handling covers this; confirm it still fires with the framer in the path.
- Overload still returns 503 before any body bytes are sent.

Update `tests/test_streaming.py` for the new SSE `info` field. It is the only test asserting the SSE
shape, and there is no compatibility window to preserve.

## Risks

- **Silent bit-depth regression.** Highest-consequence risk, because the failure mode is a
  plausible-looking file that meters at half duration. Mitigated by the header assertion test.
- **Buffered/streamed equivalence.** If concatenating the streaming source does not reproduce
  `generate()`, Exit A must keep the old call path. The test above is what tells you.
- **Streaming WAV headers.** Unknown-length RIFF is widely but not universally accepted. glyph-gate is
  the only consumer — verify against httpx and whatever it hands audio to downstream. Prefer `pcm`
  internally if `wav` proves awkward.
- **Disconnect cleanup.** Lower risk than with a subprocess encoder, since framing is pure in-process
  byte manipulation, but the framer still needs finalizing or discarding on teardown rather than only
  stopping the generator.

## Acceptance Criteria

- One audio source feeds all three exits; no exit calls `model.generate()` directly.
- Every exit emits 16-bit LE integer PCM, asserted at the header level.
- Exit A returns half the bytes of the Phase 0 baseline for identical audio duration.
- Exit B delivers first bytes before synthesis completes, and byte-derived `audio_seconds` matches the
  server-side metric within one frame.
- Unrecognized `response_format` values return 400.
- `pytest tests/` passes with no server for unit tests, and fully against a running server.
- Docs updated: `docs/API_README.md` and `docs/STREAMING_API.md` gain `response_format`, the
  little-endian contract, and the `X-Audio-*` headers, and **lose** the non-existent
  `streaming_chunk_size`, `streaming_strategy`, and `streaming_quality` parameters.
- `CHANGELOG.md` records two behaviour changes: non-streaming responses move from 32-bit float to
  16-bit int, and unrecognized `response_format` values now error.

## Coordination

glyph-gate is unaffected by Phase 1 on its own — it sends no `stream_format` and receives Exit A, whose
shape is unchanged apart from bit depth. It passes bytes through opaquely, so this should be invisible,
but confirm before shipping.

The glyph-gate switch to streaming plus byte-derived metering is Phase 4, and has its own ordering
constraint: metering must move to byte-derived **before** the streaming transport becomes its default,
or usage data goes missing in between.
