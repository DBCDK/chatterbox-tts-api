# Phase 0 — Baseline And Safety Net

Part of the streaming redesign. See `prompts/streaming-redesign-plan.md` for the overall plan and the
resolved design decisions. This document covers Phase 0 only.

## Context

The service wraps Chatterbox TTS behind `POST /v1/audio/speech`. Upcoming phases will:

- restructure the wire protocol and standardize the sample format (Phase 1)
- replace the archived `coral_chatterbox` dependency with pinned upstream `ResembleAI/chatterbox`,
  reimplementing text normalization and sentence splitting locally (Phase 2)
- introduce token-level streaming inside each sentence (Phase 3)

All three can regress audio quality in ways ordinary tests will not catch. Phase 2 risks
text-preprocessing drift; Phase 3 risks audible artifacts at chunk boundaries; Phase 1 changes the
bit depth of the non-streaming response.

## Objective

Produce the measurements and the regression harness that later phases are judged against.

**Phase 0 changes no production behaviour.** The only source change is a test-infrastructure fix
(Deliverable 4).

## Why This Must Come First

One deliverable is genuinely unrecoverable later: the **text-preprocessing snapshot**. It records what
coral's normalizer and splitter produce for our corpus. Once the dependency is swapped in Phase 2, the
old behaviour cannot be regenerated without reinstalling an archived package. Capture it while the
current stack is still installed.

The performance baseline is also cheaper now than later — after Phase 1 there is no unmodified
reference to compare against.

## Deliverable 1 — Reference Corpus

### What

A fixed set of Danish input texts, committed to the repo as plain data.

Suggested location: `tests/audio_quality/corpus.json` — a list of `{id, text, category, why}` entries.
The `why` field matters; a corpus nobody understands gets deleted in a year.

### Coverage

Aim for ~20 entries spanning:

| Category | Why it is in the corpus |
|---|---|
| Short single sentence | Fastest signal; smallest diff surface |
| Single character / 2 chars | Exercises the `AlignmentStreamAnalyzer` IndexError guard patched in `app/core/chatterbox_patches.py` |
| The lokalplan text | Real production workload; ~1200 chars, ~11 sentences. Source: `glyph-gate/scripts/test_tts_parallel.sh` |
| Danish numerals | `1.000,50`, `3,14`, `2026`. Period-as-thousands and comma-as-decimal is the case the normalizer exists for |
| Ordinals and dates | `1. januar`, `3. udgave` — abbreviation-vs-sentence-boundary ambiguity for the splitter |
| Danish abbreviations | `bl.a.`, `f.eks.`, `ca.`, `m.v.` — each contains a period that is **not** a sentence end |
| Æ/Ø/Å and casing | Tokenizer coverage |
| Long text at `MAX_TOTAL_LENGTH` | 3000 chars; upper bound behaviour |
| Text with em-dash trailing content | The normalizer strips `\s*-{2,}.*$`; verify intent |
| Mixed Danish/English | The deployed model is configured `da,en` |

The abbreviation cases are the highest-value entries. Sentence splitting is what the whole streaming
design depends on, and `f.eks.` mid-sentence is precisely where NLTK punkt gets it wrong.

### Reference Audio Storage

**Do not commit WAVs.** Generate reference audio into a gitignored directory on demand and commit only
derived metrics (see Deliverable 3). This avoids git-lfs entirely and sidesteps the fact that the
committed bytes would become stale the moment Phase 1 changes the bit depth.

## Deliverable 2 — Text Preprocessing Snapshot

### What

For every corpus entry, capture what the **current** stack produces before any audio is generated:

```json
{
  "id": "abbrev-01",
  "input": "Planen omfatter bl.a. friarealer og beplantning m.v.",
  "normalized": "...",
  "sentences": ["...", "..."]
}
```

Committed as `tests/audio_quality/preprocessing_baseline.json`.

### How

Call the current implementation directly — not through the HTTP API, which does not expose intermediate
text:

```python
from chatterbox.utils.normalizer import normalize_text
from chatterbox.utils.splitter import split_sentences
```

Use `language="da"` and mirror the runtime config: `Config.NORMALIZE_TEXT` and the `sentence_split=True`
/ `inter_sentence_silence_ms=100` values hardcoded at `app/core/tts_model.py:205`.

### Why It Matters

This is the Phase 2 acceptance gate. Comparing *text* isolates preprocessing changes from model
changes — if you only compare waveforms, a normalizer difference and a tokenizer difference look
identical, and you will spend a day bisecting. A diff on this file names the cause immediately.

## Deliverable 3 — Performance And Format Baseline

### Environment

Run on the target NVIDIA GPU (ai-p301), not a dev machine. Per the resolved device question, GPU is
the only environment where numbers mean anything.

Record alongside the results: GPU model, driver, `MODEL_INSTANCE_COUNT`, `MODEL_REPO_ID`, model
revision, and the commit SHA under test.

### Metrics To Capture

The service already instruments everything needed. Scrape `GET /metrics` before and after each run:

| Metric | What it gives you |
|---|---|
| `chatterbox_tts_time_to_first_chunk_seconds` | The number Phase 3 must improve |
| `chatterbox_tts_request_duration_seconds` | End-to-end |
| `chatterbox_tts_generation_duration_seconds` | Excludes lease wait |
| `chatterbox_tts_lease_wait_seconds` | Queue pressure at concurrency 24 |
| `chatterbox_tts_audio_seconds` | Combined with generation duration, gives realtime factor |
| `chatterbox_tts_chunk_count` | Confirms the current one-chunk-per-sentence granularity |

Also capture VRAM per instance from `GET /health` (`memory_info`, `pool_status`).

### Runs

1. **Concurrency 1**, non-streaming, whole corpus. The clean per-request picture.
2. **Concurrency 1**, `stream_format=sse`, whole corpus. Gives current TTFC.
3. **Concurrency 24**, lokalplan text. Saturation behaviour — the pool is 4 instances, so this is
   oversubscribed sixfold. Use `glyph-gate/scripts/test_tts_parallel.sh` with `REQUEST_COUNT=24`.

Report **p50 and p95**, not means. Histogram buckets are already defined in `app/core/metrics.py:38`.

### Format Facts To Record

The two paths currently disagree on bit depth, and Phase 1 standardizes them. Record the starting
point so the change is measured against a fact rather than a memory:

| Path | Expected | Verify with |
|---|---|---|
| Non-streaming | 32-bit float WAV, `audioFormat: 3`, 96,000 B/s | `ffprobe -show_streams speech.wav` |
| SSE | 16-bit int PCM, 48,000 B/s | Decode a delta and check byte count against frame count |

Record total response bytes for the corpus on each path. Phase 1's non-streaming response should drop
by half.

### Determinism — Read Before Generating Reference Audio

Generation is **stochastic**: `TEMPERATURE=0.8`, plus `TOP_P`, `MIN_P`, and `REPETITION_PENALTY`.
Re-running the same input does not produce the same waveform. Two consequences:

1. **Reference audio must be seeded** to be worth anything. Seed immediately before each generate,
   pin to a single model instance, and set `MODEL_INSTANCE_COUNT=1` for baseline runs so thread
   scheduling cannot reorder RNG consumption.
2. **Do not assume sample-exact end-to-end comparison will survive Phase 3.** Token-level streaming
   changes the decode loop, which changes how much randomness is consumed and in what order. An
   end-to-end null test may be unachievable by construction.

This shapes Deliverable 4.

## Deliverable 4 — The Null Test Harness

### The Problem It Solves

Phase 3 carries flow-matching and HiFTGAN caches across chunk boundaries. Getting that wrong produces
clicks and phase discontinuities — audible, but easy to miss in casual listening and invisible to a
duration check.

### Design: Two Tiers

**Tier 1 — vocoder isolation (deterministic, the real gate).**

Do not compare "streamed request" against "non-streamed request". Instead:

1. Generate speech tokens once for a corpus entry and cache them.
2. Run the vocoder over those tokens two ways — whole-sequence, and chunked with cache carry-over.
3. Compare waveforms sample-by-sample.

This isolates the exact code Phase 3 changes, and it is deterministic because the token sequence is
fixed input. Note that flow matching itself samples noise, so seed it too, or pass fixed `noised_mels`.

Suggested threshold: segmental SNR > 40 dB, with **no** single sample deviating more than a small
epsilon. Chunk-boundary clicks are localized, so a global average SNR will hide them — check the
regions around boundary offsets specifically.

**Tier 2 — end-to-end statistical comparison.**

For full-request comparisons where sample-exactness is not achievable, compare:

- total duration (within one frame)
- RMS level per sentence
- a spectral fingerprint (e.g. mean log-mel per sentence, cosine distance)
- silence-gap structure between sentences (the 100 ms inter-sentence silence should still be there)

Tier 2 catches gross regressions. It is not a substitute for the listening gate.

### Human Sign-Off

Automated tiers are necessary, not sufficient. Audio quality sign-off is owned by Kristian Nørgaard
Jensen (krje@dbc.dk), against this corpus, and gates Phase 2 and Phase 3 acceptance.

## Deliverable 5 — Make Unit Tests Meaningful Locally

### The Problem

`tests/conftest.py:58` defines a **session-scoped `autouse`** fixture:

```python
@pytest.fixture(scope="session", autouse=True)
def check_api_health(api_client: APIClient):
    if not api_client.wait_for_health():
        pytest.skip(f"API not available at {BASE_URL}. Please start the server first.")
```

Because it is `autouse`, it applies to every test in `tests/` — including the pure unit tests in
`test_model_pool.py`, `test_metrics.py`, and `test_request_timeouts.py`, which build fake model objects
and never touch the network. On a machine with no server running, **the entire suite skips**, and a
skipped suite reports green.

Phases 1 and 3 add unit-testable logic (framers, chunk policy) that must be verifiable on a dev
machine per the resolved device decision. A suite that silently skips cannot serve that purpose.

### The Fix

Scope the health gate to the integration tests only. The marker machinery already exists in the same
file — `pytest_collection_modifyitems` tags `test_api` and `test_streaming` with `api`. Options, in
order of preference:

1. Drop `autouse` and request `check_api_health` explicitly in the integration test classes.
2. Keep `autouse` but make it a no-op unless the requesting test carries the `api` marker.
3. Move integration tests under `tests/integration/` with their own conftest.

Whichever is chosen, verify afterwards that `pytest tests/test_model_pool.py` **passes** — not skips —
with no server running.

## Acceptance Criteria

- `tests/audio_quality/corpus.json` committed, ~20 entries, each with a stated rationale.
- `tests/audio_quality/preprocessing_baseline.json` committed, generated from the **current** coral
  stack. This is the irreplaceable artifact.
- Baseline numbers recorded in this document (append a `## Results` section) with p50/p95, environment
  details, and the commit SHA measured.
- Current bit depth and bytes-per-second confirmed for both paths and written down.
- Null test harness runs in CI and passes against the current stack. Tier 1 must be able to fail —
  prove it by deliberately corrupting a chunk boundary and confirming a red result.
- `pytest tests/test_model_pool.py` passes with no server running.

## Out Of Scope

- Any change to `app/api/endpoints/speech.py`, the request model, or the response format.
- The `response_format` parameter, framing, and headers — all Phase 1.
- Touching the `coral_chatterbox` dependency — Phase 2.
- CUDA graphs / `generate_fast`. Already tried and rejected; see the parent plan.

## Results

Measured 2026-07-28 against `ai-p301`. Commit under test: `055e4daf0154160c594289ba6294a570ae09a40e`
(HEAD of `main` at measurement time) — confirmed by Kristian as what the running container was built
from.

**Caveat: `ai-p301` is a dev box, not representative of production topology.** It has 4x L40S GPUs (the
service uses only 1) and was configured with 5 model instances for this measurement. Production runs a
single L40S GPU but packs **12 model instances** onto it. This matters most for Run 3 below: 24
concurrent requests against 5 instances is ~4.8x oversubscription; against production's 12 instances
it's only ~2x. The "zero of 24 requests succeeded" result almost certainly does not generalize to
production as-is — it should be re-measured with `MODEL_INSTANCE_COUNT=12` before drawing conclusions
about production saturation behavior.

### Environment

| Field | Value |
|---|---|
| GPU model | NVIDIA L40S (46,068 MiB) — box has 4x L40S, but the service only uses GPU 0 |
| Driver / CUDA | Driver 610.43.02, CUDA (UMD) 13.3 |
| `MODEL_INSTANCE_COUNT` | **5** (the plan's "pool is 4 instances" assumption is stale — see below) |
| `MODEL_SOURCE` / `MODEL_REPO_ID` | `local_dir`, resolved path `/cache/roest-v3-chatterbox-500m` (`MODEL_REPO_ID` unset) |
| Model revision | not exposed by `/health`; unknown |
| Commit SHA | `055e4daf0154160c594289ba6294a570ae09a40e` |
| GPU memory (idle, 5 instances loaded) | 16.8 GB allocated / 16.85 GB reserved (of 46 GB available on GPU 0) |

### Format Facts

| Path | audioFormat | bits/sample | byte rate | Matches doc expectation? |
|---|---|---|---|---|
| Non-streaming (`stream_format` omitted/`"audio"`) | 3 (IEEE float) | 32 | 96,000 B/s | Yes, exactly |
| SSE (`stream_format="sse"`) | n/a (raw PCM deltas) | 16 (int) | 48,000 B/s | Yes, exactly |

Confirmed via WAV header inspection (`audioFormat=3, channels=1, sample_rate=24000, bits_per_sample=32`)
and by decoding an SSE delta (`speech.audio.info` declares `sample_rate=24000, bits_per_sample=16`; a
single-sentence delta was 90,240 bytes for one chunk, evenly divisible by 2 bytes/sample as expected).

### Run 1 — Concurrency 1, non-streaming, whole corpus (n=20)

19/20 succeeded. **`char-01` ("A", 1 char) got HTTP 400** — the deployed config enforces
`min_text_length: 2`, so this corpus entry's stated purpose (exercising the `AlignmentStreamAnalyzer`
IndexError guard) cannot actually be exercised through the HTTP API; it would need a direct model call
like Deliverable 2 uses. Worth a note in `corpus.json` or a follow-up when Phase 1/3 touch validation.

| Metric | p50 | p95 |
|---|---|---|
| Request duration (client-measured, n=19 successes) | 1.65 s | 31.8 s (dominated by the 3000-char `maxlen-01` entry at 73.1 s) |
| Server-recorded `request_duration_seconds` (histogram, n=20) | ~1.70 s | ~30 s |

Total response bytes for the corpus (19 successes): **32,307,440 bytes**.

### Run 2 — Concurrency 1, `stream_format=sse`, whole corpus (n=20)

19/20 succeeded (same `char-01` 400). Time-to-first-chunk: server-recorded mean ≈ 1.67 s
(`sum=31.674s / count=19`); p50/p95 request duration were close to Run 1 (1.81 s / 31.5 s
client-measured) since most corpus entries are single-sentence (1 chunk).

Total response bytes for the corpus (19 successes): **21,333,630 bytes** — about 66% of the
non-streaming total, consistent with 16-bit vs 32-bit samples plus SSE's JSON/base64 framing overhead
eating back some of the raw halving.

### Run 3 — Concurrency 24, lokalplan text, 5-instance pool (`tests/tts_presure_test.sh`, `REQUEST_COUNT=24`)

**Correction to the plan's assumption:** the pool on this dev box is 5 instances, not 4, so this is
~4.8x oversubscription, not sixfold. **This is also the dev-box caveat above at its sharpest: production
runs 12 instances, so this same 24-request batch would be only ~2x oversubscribed there, not ~4.8x — the
result below should not be read as "this is what production does."** The result on this box was more
dramatic than "slower":

- **19/24 requests got HTTP 503** at ~60.0–60.1 s each — exactly `MAX_QUEUE_WAIT_SECONDS=60.0`. The
  server's own metrics confirm these as `outcome="overload"` (count went from 0 → 19).
- **5/24 requests got HTTP 504** at ~341–345 s wall-clock (client-side) — these got a lease eventually
  but the server recorded them as `outcome="timeout"` (count 0 → 5), and something in front of the app
  (a gateway/proxy) timed out the client connection well past the app's own `REQUEST_TIMEOUT_SECONDS=120`.
- **Zero of the 24 concurrent requests succeeded end-to-end.**

This is the key saturation finding *for a 5-instance pool*: at ~4.8x oversubscription, the current pool
doesn't degrade gracefully into higher latency — it fails outright for every request in the batch.
Whether this holds at production's 12-instance/~2x oversubscription is unmeasured; re-run with
`MODEL_INSTANCE_COUNT=12` before treating this as a production risk. Time-to-first-chunk
improvements from Phase 3 will help perceived latency under *moderate* load, but won't by themselves
fix this — it's a queue-capacity/admission-control problem, not a generation-latency problem.

### Null Test Harness (Deliverable 4)

Both tiers implemented under `tests/audio_quality/`, gated behind `RUN_AUDIO_QUALITY_TESTS=1` (they load
real model weights and take 10–400s; excluded from the default fast unit-test run). Both pass against
the current stack, and both have an automated (not just manual) proof they can fail:

- **Tier 1** (`vocoder_isolation.py` / `test_null_harness.py`): generates speech tokens once for a fixed
  corpus entry, then compares whole-sequence vocoder synthesis against chunked synthesis at every chunk
  boundary. Two real findings from building this:
  1. The flow-matching stage's `finalize=False` streaming hook is currently broken in the installed
     `chatterbox` fork — a shape-mismatch bug in `flow.py`'s masking when `h` is look-ahead-trimmed but
     `h_lengths` isn't. True incremental (token-level) mel computation isn't usable via the public API
     today; this is worth flagging before Phase 3 planning assumes it works.
  2. HiFTGAN's `cache_source` "avoid glitch" mechanism *alone* does not prevent chunk-boundary
     discontinuities (tested both ways: near-identical, large boundary error regardless of whether it's
     wired up). What actually works, using only currently-functional APIs: **overlap-and-trim** — decode
     each mel chunk plus ~10 frames of context on each side, keep only the center. This dropped boundary
     deviation by 50-100x (from >100% of peak amplitude down to ~1-3%). Phase 3's HiFTGAN-stage chunking
     should plan around this, not around `cache_source` alone.
- **Tier 2** (`statistical_comparison.py` / `test_statistical_comparison.py`): duration, per-sentence
  RMS, mean-log-mel cosine distance, and silence-gap structure, exercised against the multi-sentence
  lokalplan text via the production `ChatterboxInference.generate()` path (same class `app/core/tts_model.py`
  uses). Self-comparison (same seed, two independent generations) passes tightly; a deliberately
  silence-stripped variant is reliably caught by the silence-gap-structure check.

### Deliverable 5 confirmation

`pytest tests/test_model_pool.py` passes 12/12 with no server running (~2s, no health-check skip).

Full suite also re-run with `CHATTERBOX_TEST_URL=http://ai-p301:4123` so the integration tests
(`test_api.py`, `test_streaming.py`) exercise a live server instead of skipping: **37 passed, 4 skipped**
(the 4 skips are the opt-in `RUN_AUDIO_QUALITY_TESTS` tests from Deliverable 4, not integration tests).
