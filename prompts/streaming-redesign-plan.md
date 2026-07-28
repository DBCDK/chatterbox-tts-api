# Streaming Redesign And Upstream Migration Plan

Status: **all questions resolved.** Phase 0 and Phase 1 have detailed plans of their own:

- `prompts/streaming-phase-0-plan.md` — baseline measurements and regression harness
- `prompts/streaming-phase-1-plan.md` — wire protocol, framing, sample-format standardization

This document remains the source of truth for cross-phase decisions and rationale.

## Decisions

All nine opening questions have been answered. Kept in place rather than deleted so the reasoning
behind each choice stays with the plan.

Summary:

| # | Topic | Decision |
|---|---|---|
| Q1 | Licensing | No coral code copied; text utils reimplemented locally |
| Q2 | Fork vs migrate | Migrate to pinned upstream `ResembleAI/chatterbox` |
| Q3 | Opus | Out — same-cluster traffic, `pcm` + `wav` only |
| Q4 | CUDA graphs | Out of scope; `generate_fast` already tried and rejected |
| Q5 | Device matrix | NVIDIA GPU only; CPU/MPS for unit tests |
| Q6 | Usage metering | glyph-gate derives `audio_seconds` from byte count |
| Q7 | OpenAI parity | `wav` default retained, divergence documented |
| Q8 | Quality sign-off | Kristian Nørgaard Jensen (krje@dbc.dk) |
| Q9 | SSE compatibility | No compat window; SSE becomes a debug-only path |

### Q1 — Licensing of the vendored coral code — RESOLVED

**Decision: no coral code is copied.** The text utilities are reimplemented locally, and the
CUDA-graph modules are out of scope (Q4). Nothing carrying the fork's **OpenRAIL-S** license enters
this **AGPLv3** repo, and upstream `ResembleAI/chatterbox` is MIT.

One implementation constraint follows from this, and it belongs in the Phase 2 work rather than as an
open question:

- The reimplementation must be written from **requirements**, not by transcribing coral's structure.
  The riskiest parts to carry across are the language tables (`_NUM2WORDS_LANGS`,
  `_PERIOD_DECIMAL_LANGS`, `_NLTK_LANGUAGE_MAP`) and the specific regexes, since those are the
  copy-shaped expression rather than the underlying idea.
- Scoping to the languages we actually serve handles this naturally. `.env.example:32` configures
  `MODEL_SUPPORTED_LANGUAGES=da,en`, so a da/en normalizer needs one decimal convention (comma) and
  one nltk language, with pass-through for anything else. Fewer lines, genuinely different shape,
  and testable against real Danish input instead of a 20-language table nobody exercises.

Note on model weights: unaffected either way. `CoRal-project/roest-v3-chatterbox-500m` and the
upstream ResembleAI weights carry their own licenses independent of which Python package loads them.
Not a factor in this migration.

### Q2 — Fork the fork instead? — RESOLVED

**Decision: migrate to pinned upstream `ResembleAI/chatterbox`.** We do not fork
`alexandrainst/coral_chatterbox` into `DBCDK/`.

### Q3 — Is Opus in scope on the raw-audio transport? — RESOLVED

**Decision: no. Ship `pcm` and `wav` only.**

Both services run in the same k8s cluster, so audio never crosses a constrained link. Pod-to-pod
traffic over the CNI at 384 kbit/s per stream, capped at `MODEL_INSTANCE_COUNT=4` concurrent streams,
is roughly 1.5 Mbit/s aggregate — irrelevant on a cluster network.

Two consequences worth recording:

- **No encoder in the request path.** No ffmpeg subprocess or PyAV dependency, and none of the
  associated lifecycle and orphan-process risk. The encoder seam in Phase 1 stays as an interface so
  `opus` can be added later without a contract change, but the only implementations are PCM
  passthrough and a streaming WAV header.
- **If compression is ever needed for the glyph-gate-to-client hop, it belongs in glyph-gate, not
  here.** That hop leaves the cluster and may genuinely warrant Opus, but glyph-gate is the right
  place to do it: its pods are cheap and horizontally scalable, whereas encoding here would put
  ffmpeg processes on the GPU node competing for CPU with inference. Keep expensive nodes doing
  inference.

### Q4 — Are CUDA graphs actually worth porting? — RESOLVED

**Decision: out of scope.** Switching to the `generate_fast` path was tried against the current stack
and caused enough problems that fixing them is not worth it now. Combined with the fact that the path
is not used in production today, there is no case for porting ~300 lines of static-KV-cache code and
integration hooks across four files.

See *Tried And Rejected* below. If this is revisited, the `AlignmentStreamAnalyzer` interaction is
the first thing to understand — the fast path disables it, and it is the component whose `IndexError`
`app/core/chatterbox_patches.py` already patches.

### Q5 — Device matrix — RESOLVED

**Decision: NVIDIA GPU is the only target for real testing. CPU/MPS need only run unit tests.**

No device-fallback machinery is required in Phase 3 — chunked flow-matching and HiFT cache handling
are validated on GPU only, and audio quality on CPU/MPS is not a concern.

The one constraint this imposes: **the streaming code must remain importable and unit-testable without
CUDA.** The existing suite already establishes the pattern — `tests/test_model_pool.py`,
`tests/test_metrics.py`, and `tests/test_request_timeouts.py` all use fake model objects exposing
`generate_stream_async`. Keep the model interface narrow enough that fakes remain viable, and keep
CUDA-specific imports out of module top-level scope.

### Q6 — Usage metadata on the raw-audio path — RESOLVED

`audio_seconds` **is** used for metering, and glyph-gate is being updated to record it.

**Decision: no HTTP trailers, and no requirement to use SSE. glyph-gate derives `audio_seconds` from
the byte count it received.**

This works because Q3 removed compressed formats. `pcm` and `wav` have a fixed bytes-to-duration
relationship, and every term is known before generation starts:

```
audio_seconds = payload_bytes / (sample_rate * channels * bytes_per_sample)
                              = payload_bytes / 48000     # 24 kHz mono int16
```

Implementation:

- The service sends `X-Audio-Sample-Rate`, `X-Audio-Channels`, and `X-Audio-Bits-Per-Sample` as
  ordinary response headers on the `audio` transport, alongside the existing `X-Request-ID` and
  `X-Usage-Input-Chars`. All are known up front.
- glyph-gate counts bytes as it streams and divides. For `wav` it subtracts the header length.
- The server-side `chatterbox_tts_audio_seconds` histogram and the structured request log remain the
  reconciliation source, correlated by `X-Request-ID`.

Two things this buys beyond avoiding trailers:

- **No proxy-stripping risk.** Trailer support across k8s ingress and service meshes is inconsistent,
  and a silently dropped trailer would mean silently lost metering.
- **Truncated streams meter correctly.** On client disconnect, glyph-gate bills what was actually
  delivered rather than what the server intended to produce. That is the more defensible number.

SSE keeps its `speech.audio.done` usage event for callers that want it in-band, but glyph-gate does
not need SSE to meter.

### Q7 — OpenAI parity strictness — RESOLVED

An earlier draft of this plan claimed OpenAI restricts streaming to `pcm`/`wav`. That is **not** an
OpenAI constraint — it comes from vLLM-Omni, a third-party reimplementation. See *The OpenAI Contract*
below for the verified spec.

**Decision: `wav` stays the default, and the divergence from OpenAI's `mp3` default is documented.**
Adopting `mp3` would put a lossy encoder in the path of every non-streaming request for compatibility
nobody relies on.

The LiteLLM sub-question is moot: per Q9, only glyph-gate can reach the TTS server, so anything
routing through LiteLLM sits in front of glyph-gate rather than between glyph-gate and this service.

### Q8 — Quality sign-off — RESOLVED

**Owner: Kristian Nørgaard Jensen (krje@dbc.dk).**

Both the upstream migration and token-level streaming can regress audio quality in ways automated
tests will not catch — preprocessing drift in Phase 2, chunk-boundary artifacts in Phase 3. Phase 0
builds the reference corpus; sign-off against it is a listening gate on Phase 2 and Phase 3
acceptance, not a formality.

### Q9 — Compatibility window for the current SSE shape — RESOLVED

**Decision: no compatibility window. Change the shape outright.**

glyph-gate is the only caller and no other service has network access to the TTS server.
`tests/test_streaming.py` is the only thing asserting the current event shape, and it gets rewritten
with the change. No versioning, no deprecation period, no dual-shape support.

Note the follow-on: with glyph-gate metering from bytes (Q6) and streaming over `audio`, **nothing
will use SSE in production.** It stays because it is already built, it is the only mode that can report
a mid-stream error in-band, and it is useful for debugging by hand with `curl`. But it should be
treated as a secondary path — do not spend Phase 1 effort perfecting its event schema, and drop it
outright if it becomes a maintenance burden.

---

## Goal

Replace the current SSE-only streaming path with a format-negotiated streaming API, move off the
archived `coral_chatterbox` fork onto pinned upstream `ResembleAI/chatterbox`, and reduce
time-to-first-audio from "one full sentence" to "a fraction of a sentence".

Three separable outcomes, in dependency order:

1. **Wire protocol** — `stream_format` x `response_format` (`pcm`/`wav`), with a real raw-audio
   streaming mode and byte-derived usage metering.
2. **Dependency** — off the archived fork, keeping normalization and sentence splitting,
   reimplemented locally.
3. **Latency** — token-level streaming inside each sentence.

## The OpenAI Contract

Verified against the current API reference (see *Sources*), not from memory. An earlier draft of this
plan misattributed a vLLM-Omni restriction to OpenAI; this section is the corrected record.

| Parameter | Allowed values | Default |
|---|---|---|
| `response_format` | `mp3`, `opus`, `aac`, `flac`, `wav`, `pcm` | `mp3` |
| `stream_format` | `sse`, `audio` | not streaming |
| `speed` | 0.25 – 4.0 | 1.0 |
| `input` | max 4096 characters | — |

The only documented constraint is that `sse` is unsupported on `tts-1` / `tts-1-hd`.
`response_format` and `stream_format` are otherwise **orthogonal** — nothing in the spec forbids
Opus over either transport. OpenAI's own guidance nonetheless recommends `wav`/`pcm` for lowest
latency.

Not verified: the exact field schema of `speech.audio.delta` / `speech.audio.done` beyond "type plus a
base64 audio chunk" and a usage object. The reference pages for those events were not retrievable.
Worth confirming before finalizing the SSE event shape in Phase 1.

### Format And Transport Matrix

Ecosystem practice among OpenAI-compatible TTS servers:

| Implementation | Opus | SSE | Opus over SSE |
|---|---|---|---|
| openedai-speech | yes, ffmpeg `-f ogg -c:a libopus` | not documented | no |
| Kokoro-FastAPI | yes | not documented | no |
| vLLM-Omni | not on streaming paths | yes | no — restricts SSE to `pcm`/`wav` |

**Decision for this service (per Q3):** `pcm` and `wav` on both transports. No `opus`, no `mp3`,
`aac`, or `flac`.

Had compressed formats been in scope, they would have gone on `stream_format=audio` only — SSE deltas
carry base64 audio, so an Ogg/Opus container would have its pages fragmented across JSON envelopes,
paying base64's +33% on top of a format chosen for its size while forcing the client to reassemble
container boundaries before decoding. Recorded here because it is the right shape if `opus` is ever
added for a non-cluster consumer.

## Non-Goals

- WebSocket or WebRTC transports. Neither is justified by the request/response usage pattern.
- A job-and-resource API (`POST` returns id, `GET` streams). Revisit only if retry/resume becomes a
  requirement.
- Multi-voice or voice cloning per request. `resolve_voice_path_and_language` stays as-is.
- Raising `MAX_TOTAL_LENGTH`. Separate decision, separate risk.
- Changing the model pool's lease semantics. Explicitly rejected in favour of token-level streaming.
- **CUDA graph acceleration.** See *Tried And Rejected*.

## Tried And Rejected

Recorded so these are not retried without new information.

### Sentence-level model-pool fan-out

Synthesizing sentences in parallel across the 4-instance pool while streaming in order. Rejected on
two grounds:

- It optimizes the wrong regime. `glyph-gate/scripts/test_tts_parallel.sh` drives 24 parallel
  requests at a 4-instance pool — saturated sixfold. Fan-out cannot create GPU capacity; under
  saturation it reorders the queue and worsens p99.
- It can deadlock. Four requests each holding one lease and waiting for a second block until
  `MAX_QUEUE_WAIT_SECONDS` (60s) trips them. Avoiding that needs all-or-nothing acquisition, which
  harms admission, or opportunistic non-blocking acquisition, which makes latency
  non-deterministic. Either way a simple correct lease model becomes a tricky one.

Token-level streaming (Phase 3) addresses the same latency goal at `1 lease : 1 request`.

### CUDA graphs / `generate_fast`

Switching to the fork's `generate_fast` / `generate_stream_fast_async` path was tried against the
current stack and caused enough problems that fixing them was not worth it. The path is also unused
in production today, so there is no measured baseline to defend porting it.

If revisited, start with the `AlignmentStreamAnalyzer` interaction: the fast path constructs a
separate `_patched_model_fast` with the analyzer disabled, so pathological inputs can run to
`max_new_tokens` instead of terminating — and that analyzer is the component
`app/core/chatterbox_patches.py` already patches an `IndexError` in. Recording the specific failures
seen would help whoever picks this up, if they are still to hand.

## Current State

- `POST /v1/audio/speech` supports `stream_format` of `audio` or `sse`.
- `audio` is **not** streaming: it buffers the whole tensor, writes a complete WAV, then copies the
  buffer again (`io.BytesIO(buffer.getvalue())`, `app/api/endpoints/speech.py:682`).
- `sse` emits base64 **raw headerless int16 PCM**, native-endian, with format described once in a
  non-standard `speech.audio.info` event. Base64 costs a flat +33%.
- **The two paths disagree on bit depth.** Non-streaming returns 32-bit float WAV (`ta.save` preserves
  the model's float32); SSE downconverts to int16. Same audio, 2x the bytes on the non-streaming path.
  See *Sample Format* under Phase 1.
- Chunk granularity is one **sentence** — `ChatterboxInference.generate_stream_async` yields one
  fully-synthesized tensor per sentence. Time-to-first-chunk equals full synthesis of sentence one.
- `coral_chatterbox` is archived. Its only substantive addition over upstream is commit `469d62e`:
  the `ChatterboxInference` wrapper, `utils/normalizer.py`, `utils/splitter.py`, and CUDA-graph
  acceleration. Everything else (Turbo, GPT-2 config, 2454 multilingual vocab) is upstream.
- The deployed model `CoRal-project/roest-v3-chatterbox-500m` is a standard upstream multilingual-v2
  fine-tune (vocab 2454, stock filenames). Upstream loads it as-is — we are not locked in.
- The only consumer, glyph-gate's `ChatterboxTTSAdapter`, is fully blocking and never sets
  `stream_format`.
- `docs/STREAMING_API.md` and `docs/API_README.md` document `streaming_chunk_size`,
  `streaming_strategy`, and `streaming_quality` — none of which exist in `TTSRequest`.

## End State

**One production pipeline with three exits.** Today the streaming and non-streaming paths call
different model methods (`generate` vs `generate_stream_async`) and diverge from the start. After this
work they share a single source, and the only difference is how bytes leave the process.

1. A request arrives with `response_format` (`pcm` or `wav`) and optionally `stream_format`.
2. Validation, language resolution, and lease acquisition as today — unchanged, including the
   pre-stream lease acquisition that makes overload a clean 503.
3. Text is normalized and split into sentences (for quality, not latency).
4. Each sentence is synthesized with token-level streaming, yielding sub-sentence audio chunks.
5. Chunks pass through a framer chosen by `response_format`.
6. Bytes leave via one of three exits:
   - **no `stream_format`** — buffered into a complete WAV with a correct header and exact
     `X-Usage-*` headers. OpenAI's default shape. Unlike today, produced by draining the same
     streaming source rather than a separate `generate()` call, so the double buffer copy is gone.
   - **`stream_format=audio`** — chunked HTTP body with `X-Audio-*` headers. The production path.
   - **`stream_format=sse`** — base64 deltas plus a `speech.audio.done` usage event. Debug only.
7. glyph-gate streams via `audio` and meters `audio_seconds` from the byte count and the `X-Audio-*`
   headers.
8. The service depends on pinned upstream chatterbox plus local text utilities, with no dependency on
   the archived coral fork.

Collapsing to one source is a significant part of the value here: quality changes, timeout handling,
disconnect handling, and metrics stop needing to be implemented and tested twice.

## Phase 0 — Baseline And Safety Net

### Objective

Produce the numbers that gate later decisions, and the harness that detects quality regressions.
Nothing in this phase changes production behaviour.

### Changes

- Benchmark on the target GPU (ai-p301) using the lokalplan text from
  `glyph-gate/scripts/test_tts_parallel.sh`, at concurrency 1 and 24:
  - time to first chunk, total request time, realtime factor, VRAM per instance
  - existing histograms already cover most of this (`chatterbox_tts_time_to_first_chunk_seconds`,
    `app/core/metrics.py:177`)
- Build `tests/audio_quality/`:
  - a corpus of ~20 Danish texts (short, long, numerals, abbreviations, edge punctuation)
  - reference WAVs generated from the current stack, committed via git-lfs or stored out-of-band
  - a **null test**: concatenated streamed output vs `generate()` output on the same seed, compared
    by sample-level SNR and duration. This is the gate for Phase 3.

### Acceptance Criteria

- Documented baseline numbers checked into this plan.
- Null-test harness runs in CI against the current stack and passes.

## Phase 1 — Wire Protocol And Framing Seam

### Objective

Ship the API-shape and correctness improvements with zero model-layer risk, and establish the seam
that Phases 2 and 3 plug into.

Note that with Q3 resolved, bandwidth is **not** the motivation here — in-cluster bytes are free. The
value is a genuine streaming `audio` transport, removal of the double buffering, an explicit format
contract, and the seam itself.

### Key Decision

Introduce an explicit boundary between *audio production* and *byte framing*:

```
AudioSource:  AsyncIterator[torch.Tensor]  +  sample_rate
      |
      v
Framer:       bytes-in, bytes-out, streaming, with finalize()
      |
      v
Transport:    chunked HTTP body  |  base64 SSE deltas
```

This seam is what makes the later phases cheap: Phase 2 swaps the `AudioSource` implementation,
Phase 3 changes its chunk granularity, and neither touches framing or transport. It is also where an
`opus` encoder would slot in later without a contract change.

### Changes

- `app/models/requests.py`: add `response_format` with `pcm | wav`. Reject other OpenAI-documented
  values (`mp3`, `opus`, `aac`, `flac`) with a clear 400 rather than silently returning WAV, which is
  today's behaviour. Keep ignoring `speed`, but say so explicitly rather than implying support.
- New `app/core/audio/framing.py`. **Both framers emit 16-bit signed little-endian integer PCM.**
  See *Sample Format* below — this is a deliberate change to the non-streaming response, not just a
  restatement of current behaviour.
  - `PcmFramer` — int16 little-endian, no header. Today's native-endian behaviour is an unstated
    contract; document it.
  - `WavStreamFramer` — a hand-written RIFF header (`audioFormat: 1`, `bitsPerSample: 16`) with
    unknown-length size fields, then raw frames. Do **not** use `torchaudio.save` — see below. Note
    that some strict parsers reject unknown-length headers; ffmpeg/ffplay/browsers accept them.
- New `app/api/endpoints/speech_stream.py` (or a refactor of `speech.py`, which is 766 lines and
  already carries a lot): the two transports over a shared source+framer pipeline.
- Implement `stream_format=audio` as a genuine chunked response.
- Emit `X-Audio-Sample-Rate`, `X-Audio-Channels`, `X-Audio-Bits-Per-Sample` response headers on the
  `audio` transport — these are what let glyph-gate meter from byte count (Q6). All are known before
  generation starts.
- Serve the non-streaming response by draining the same streaming source and buffering at the edge,
  rather than calling a separate `generate()`. This is what collapses the two paths into one and
  removes the double buffer copy at `speech.py:682`.
- Delete `generate_speech_internal` (`speech.py:389`). It is dead code — exported but called nowhere,
  since the long-text backend that used it was already removed per `prompts/cleanup-plan.md`.
- `speech.audio.info` gains an explicit `format` field. Otherwise leave the SSE schema alone — per Q9
  it is a debugging path, not a production one.
- Preserve the pre-stream lease acquisition at `speech.py:633` — it is why capacity failures surface
  as a clean 503 instead of a truncated stream. Do not move it into the generator.

### Sample Format

The two current paths **disagree on bit depth**, and standardizing them is part of this phase.

Verified by running `torchaudio.save` on a float32 tensor (torchaudio 2.7.1):

| Path | Format written | Bytes/sec at 24 kHz mono |
|---|---|---|
| Non-streaming `ta.save(buffer, tensor, sr, format="wav")` | 32-bit float, `audioFormat: 3` | 96,000 |
| SSE `(clamp(t, -1, 1) * 32767).to(torch.int16)` | 16-bit int | 48,000 |

The model emits float32 and `ta.save` preserves it, so the non-streaming endpoint returns 32-bit float
WAV while SSE returns int16. Same audio, 2x the bytes, and `audioFormat: 3` is less widely supported
than integer PCM.

**Decision: 16-bit signed little-endian integer PCM everywhere.** 32-bit float carries no useful
dynamic range for speech playback and most consumers downconvert anyway.

Two consequences to guard:

- **This is a visible change to the non-streaming response.** Existing clients receive 32-bit float
  WAV today. glyph-gate passes bytes through opaquely so it should not care, but confirm before
  shipping, and note it in `CHANGELOG.md`.
- **The metering formula in Q6 depends on this.** `payload_bytes / 48000` is correct for int16 and
  silently reads **half** the true duration for 32-bit float. Anyone reaching for
  `torchaudio.save` in the WAV framer reintroduces a 2x billing error from a library default. Write
  the header by hand, and add a test asserting `bitsPerSample == 16` and `audioFormat == 1` on the
  emitted header.

### Risks

- **Client disconnect** already has handling (`ClientDisconnected`); it must now also finalize or
  discard the framer, not just stop the generator. Lower risk than it would have been with a
  subprocess encoder, since framing is pure in-process byte manipulation.
- **Silent bit-depth regression.** Covered above; the mitigation is a header assertion test, because
  the failure mode is a plausible-looking file that meters at half duration.
- **Streaming WAV headers.** The unknown-length RIFF header is widely but not universally accepted.
  Since glyph-gate is the only consumer, verify against httpx plus whatever it hands audio to
  downstream, and prefer `pcm` internally if `wav` proves awkward.

### Acceptance Criteria

- `pcm` and `wav` both decode correctly under `ffprobe` and play in a browser.
- `stream_format=audio` delivers first bytes before generation completes.
- `audio_seconds` derived from byte count matches the server-side histogram value to within one
  frame, across the Phase 0 corpus. This is the metering contract from Q6 — test it explicitly.
- Emitted WAV headers assert `audioFormat == 1` and `bitsPerSample == 16` on every exit, including the
  buffered non-streaming one.
- Unrecognized `response_format` values return 400, not silent WAV.

## Phase 2 — Migrate Off The Coral Fork

### Objective

Depend on pinned upstream chatterbox while keeping normalization and sentence splitting. CUDA graphs
are out of scope per Q4.

### Changes

- `pyproject.toml`: replace `coral-chatterbox` with `ResembleAI/chatterbox` pinned to an explicit
  commit SHA — not a branch.
- New `app/core/text/`, written locally from requirements (Q1):
  - `normalizer.py` — number-to-words for the languages we serve. Danish decimal/thousands
    conventions (`1.000,50`) are the case that matters; unknown languages pass through unchanged.
  - `splitter.py` — NLTK punkt sentence splitting. NLTK is already a dependency and the punkt
    download is already handled at build time (commit `840439b`).
  - Scope to `da` and `en` per `.env.example:32` rather than reproducing a 20-language table. Add
    languages when a model actually serves them.
- New `app/core/inference.py` — a thin wrapper replacing `ChatterboxInference`, exposing only what
  this service uses: `from_pretrained` / `from_local` / `from_model`, `prepare_conditionals`, `sr`,
  `generate`, and the streaming generator. Roughly a third of the fork's 493 lines.
- Re-validate `app/core/chatterbox_patches.py` against upstream. Both patches (Cangjie local-path
  loading, `AlignmentStreamAnalyzer` IndexError) were written against the fork — confirm each is
  still needed and still applies, and upstream the second one if so.
- Verify HF allow-patterns still match the deployed model:
  `ve.pt`, `t3_mtl23ls_v2.safetensors`, `s3gen.pt`, `grapheme_mtl_merged_expanded_v1.json`,
  `conds.pt`, `Cangjie5_TC.json`.
- `tts_model.py:205-206` currently hardcodes `sentence_split=True` and
  `inter_sentence_silence_ms=100`. Make both configurable while rewriting.

### Risks

- **Preprocessing drift** is the main quality risk. Upstream's `punc_norm` and tokenizer may have
  moved since the Danish fine-tune was trained. The Phase 0 corpus is the detector.
- **Silent behavioural differences** in `prepare_conditionals` caching. The fork tracks
  `_last_audio_prompt_path` to avoid recomputing speaker embeddings; the replacement wrapper must
  keep that or per-request latency regresses.

### Acceptance Criteria

- Phase 0 quality corpus passes against the migrated stack, with human sign-off per Q8.
- No import of `chatterbox.inference` remains in `app/`.
- Model loads from all three sources (`default`, `hf_repo`, `local`) unchanged.
- Number normalization and sentence splitting produce identical output to the current stack on the
  Phase 0 Danish corpus. This is the one place where reimplementation could silently change audio,
  so compare the *text* output directly, not just the resulting waveform.

## Phase 3 — Token-Level Streaming

### Objective

Cut time-to-first-audio from a full sentence to a fraction of one, without changing the
one-lease-per-request pool model.

### Background

The primitives exist in the model code but are unwired:

- `flow.inference(..., finalize=False)` supports non-final chunks, ignoring the last 3 tokens as
  lookahead (`models/s3gen/s3gen.py:200`).
- `hift_inference(speech_feat, cache_source)` accepts a HiFTGAN cache for waveform continuity.
- The sync path throws both away:
  `# TODO jrm: ignoring the speed control (mel interpolation) and the HiFTGAN caching mechanisms for now.`
  with `hift_cache_source` hardcoded to an empty tensor (`s3gen.py:290`), and `inference()` passing
  `finalize=True` and `None`.
- A docstring references an `S3GenStreamer` class that does not ship.

So this is wiring, not invention — but the caches are precisely what prevents audible clicks at
chunk boundaries, so it cannot be shortcut.

### Changes

- Make the T3 decode loop (`models/t3/t3.py:459`) a generator yielding speech tokens.
- Implement the streaming vocoder path: carry flow and HiFT caches across chunks, honour the
  3-token lookahead, and apply `trim_fade` **only to the first chunk** (it is currently applied
  unconditionally per call, which would fade in the front of every chunk).
- Chunk policy: N speech tokens per emission, configurable. Start around 25 tokens (~1 s) and tune
  down; smaller chunks mean lower latency but more per-chunk flow overhead.
- Compose with sentence splitting rather than replacing it. **Splitting stays** — not as a latency
  device, but as a sequence-length and quality device: `generate()` defaults to
  `max_new_tokens=1000` (~40 s) and the model is trained on utterance-length inputs. A 1200-character
  document as one T3 sequence is far outside training distribution.
- Config flag with fallback to the Phase 2 sentence-level path.
- Keep `AlignmentStreamAnalyzer` active on the streaming path. It is the hallucination guard that
  stops pathological inputs running to `max_new_tokens`, and per Q4 we are deliberately staying on
  the code path that has it.
- Per Q5, target NVIDIA GPU only — no CPU/MPS fallback machinery. But keep the streaming code
  importable without CUDA so the unit suite still runs on dev machines, following the existing
  fake-model pattern in `tests/test_model_pool.py`.

### Risks

- **Chunk-boundary artifacts** — the primary risk, caught by the Phase 0 null test.
- **Throughput, not just latency.** Chunked flow-matching does slightly more total work than one
  batched call. Measure aggregate throughput at concurrency 24, not just TTFB at concurrency 1. If
  throughput regresses materially under saturation, the flag lets us ship latency wins only for
  low-concurrency traffic.

### Acceptance Criteria

- Null test passes: concatenated streamed output matches `generate()` within the agreed SNR
  threshold.
- Time-to-first-chunk drops below 500 ms p50 on the target GPU (adjust once Phase 0 gives real
  numbers).
- Throughput at concurrency 24 is within 10% of the Phase 2 baseline.
- Feature flag defaults off until sign-off, then flips.

## Phase 4 — Consumers And Documentation

### Changes

- `glyph-gate/src/glyph_gate/adapters/adapter_chatterbox_tts.py`: switch to `httpx.stream()`,
  propagate content type and streaming body through `SpeechResponse`, and distinguish upstream errors
  that arrive before the first byte (a clean status code can still be returned) from mid-stream
  failures (the response has already started). Separate repo, separate PR.
- **glyph-gate metering** (per Q6): count streamed bytes, divide by
  `sample_rate * channels * bytes_per_sample` from the `X-Audio-*` response headers, and record
  against `X-Request-ID`. For `wav`, subtract the header length. This is the piece that must land
  before the old `X-Usage-Audio-Seconds` header stops being authoritative — sequence it carefully so
  metering is never silently absent.
- `docs/STREAMING_API.md` and `docs/API_README.md`: delete the non-existent `streaming_chunk_size`,
  `streaming_strategy`, `streaming_quality`; document `response_format` x `stream_format`, the
  little-endian PCM contract, the `X-Audio-*` headers, and the metering derivation per Q6.
- `README.md`: update the SSE example.
- `CHANGELOG.md`: note the SSE payload change and the `response_format` validation change (previously
  unrecognized values silently returned WAV).

## Sequencing Rationale

Phase 1 before Phase 2 is deliberate. The protocol work is pure app-layer, carries no model risk, and
defines the `AudioSource` seam that makes the dependency swap and the token-streaming work drop-in.
Doing the migration first would mean building the protocol on a moving foundation.

Phase 0 produces the regression detector that Phases 2 and 3 both depend on. It no longer gates a
CUDA-graph decision (Q4 closed that), so it can be scoped to the corpus, the null test, and baseline
numbers.

Phase 4's glyph-gate metering change has an ordering constraint of its own: metering must move to
byte-derived before the streaming transport becomes the default, or usage data goes missing for the
window in between.

## Files Touched

| Area | Files |
|---|---|
| Request/response models | `app/models/requests.py`, `app/models/responses.py` |
| Endpoint | `app/api/endpoints/speech.py` (766 lines — split during Phase 1) |
| New: framing | `app/core/audio/framing.py` |
| New: text utils | `app/core/text/normalizer.py`, `app/core/text/splitter.py` |
| New: inference wrapper | `app/core/inference.py` |
| Model loading | `app/core/tts_model.py`, `app/core/chatterbox_patches.py` |
| Config | `app/config.py` |
| Deps | `pyproject.toml`, `uv.lock` |
| Tests | `tests/test_streaming.py`, new `tests/audio_quality/` |
| Docs | `docs/STREAMING_API.md`, `docs/API_README.md`, `README.md`, `CHANGELOG.md` |
| Separate repo | `glyph-gate/src/glyph_gate/adapters/adapter_chatterbox_tts.py` |

## Verification Notes

Findings in this plan were verified against the working tree and the local uv/HF caches:

- coral-vs-upstream attribution via `git log` on `alexandrainst/coral_chatterbox` and a diff against
  the cached upstream `chatterbox-tts` 0.1.4 snapshot.
- Model compatibility by reading the tokenizer vocab (2454) and file list from the cached
  `CoRal-project/roest-v3-chatterbox-500m` snapshot.

Not yet verified, and worth checking before Phase 2 starts:

- The exact contents of upstream chatterbox at current HEAD. The comparison above used a v0.1.4
  snapshot from the uv cache, so upstream may have moved.
- Whether the `S3GenStreamer` referenced in the docstring exists in current upstream. If it does,
  Phase 3 shrinks considerably.
- The `speech.audio.delta` / `speech.audio.done` field schemas, per *The OpenAI Contract*.

## Sources

- OpenAI Create Speech reference:
  https://developers.openai.com/api/reference/resources/audio/subresources/speech/methods/create
- OpenAI text-to-speech guide: https://platform.openai.com/docs/guides/text-to-speech
- vLLM-Omni speech API: https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/speech_api/
- openedai-speech: https://github.com/matatonic/openedai-speech
- Kokoro-FastAPI: https://github.com/remsky/Kokoro-FastAPI
