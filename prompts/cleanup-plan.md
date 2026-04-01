# Chatterbox TTS API Cleanup Plan

## Goal

Reduce the current repo to a minimal, maintainable FastAPI service that keeps only the core text-to-speech API surface:

- `POST /v1/audio/speech`
- streaming and non-streaming support
- startup model loading
- support for default, Hugging Face repo, and local finetuned model loading
- basic health visibility
- per-response usage metadata for `input_chars` and audio duration

Everything else should be removed or folded into the remaining minimal implementation.

## Target End State

### Keep

- FastAPI app startup and error handling in `app/main.py`
- core model loading in `app/core/tts_model.py`
- one request model for speech generation in `app/models/requests.py`
- a reduced response model set in `app/models/responses.py`
- one speech endpoint module for `POST /v1/audio/speech`
- one health endpoint module
- minimal configuration in `app/config.py`
- test coverage for speech generation, health, validation, and response usage metadata

### Remove

- voice library subsystem
- voice upload endpoints
- long text job queue and persistence
- background worker lifecycle for long text
- memory inspection and memory-management endpoints
- status/history/info/config endpoints beyond a minimal health check
- endpoint alias framework except the explicit routes we keep
- frontend assets and UI
- feature-specific docs and tests for deleted behavior

## What Is Currently Bloated

### Large or coupled modules

- `app/api/endpoints/speech.py`: `1492` lines, currently mixes request handling, streaming, uploads, voice resolution, memory cleanup, SSE formatting, and in-memory status updates
- `app/api/endpoints/voices.py`: `791` lines
- `app/core/voice_library.py`: `590` lines
- `app/api/endpoints/long_text.py`: `1064` lines
- `app/core/status.py`: `299` lines

### Startup complexity that should go away

`app/main.py` currently does all of the following during lifespan:

- starts async model initialization
- initializes and restores the voice library
- starts the long-text background processor
- stops the long-text processor on shutdown

For the minimal service, startup should only initialize the model.

### Router complexity that should go away

`app/api/router.py` currently mounts:

- `speech`
- `long_text`
- `voices`
- `health`
- `models`
- `memory`
- `config`
- `status`

The reduced router should mount only:

- `speech`
- `health`

Optional:

- `models` only if OpenAI client compatibility depends on it for your intended consumers

## Cleanup Principles

1. Preserve working model loading first.
2. Preserve the current `/v1/audio/speech` behavior before deleting adjacent systems.
3. Extract the minimal speech path before removing long-text and upload callers.
4. Replace in-memory tracking with lightweight per-response usage metadata instead of trying to keep both.
5. Delete dead modules fully once references are removed.
6. Prefer smaller, explicit routing over generic alias infrastructure.

## Proposed Minimal API Contract

### Required endpoint

- `POST /v1/audio/speech`

### Optional endpoint

- `GET /health`

### Request shape

Keep a reduced version of `TTSRequest` with:

- `input`
- `voice` only if you still want OpenAI-style request compatibility, but it should map to the configured sample rather than a voice library
- `response_format` if compatibility requires it, but the implementation can still always return WAV if desired
- `stream_format`
- `exaggeration`
- `cfg_weight`
- `temperature`
- only the streaming fields that are actually needed for the chosen streaming implementation

### Response behavior

- Non-streaming: full generated audio response
- Streaming: choose one of these and standardize on it
  - raw chunked audio
  - SSE
  - both, selected by `stream_format`

The current repo supports both raw audio streaming and SSE. If external clients depend on that, keep both under the same endpoint. If not, pick one to simplify the implementation.

## Per-Response Usage Metadata Plan

This replaces the current in-memory request history and status subsystem.

### Problems with the current approach

- `app/core/status.py` stores only in memory
- history is capped and ephemeral
- it is not multi-process safe
- SSE usage accounting is currently a rough estimate, not durable usage data

### Usage requirements

For each successful request, expose:

- `input_chars`
- audio duration in seconds

Do not add persistent storage or a request history database.

### Recommended implementation

For non-streaming audio responses:

- return raw audio as before
- include usage metadata in response headers

Suggested headers:

- `X-Usage-Input-Chars`
- `X-Usage-Audio-Seconds`

For SSE streaming responses:

- include the same values in the final `speech.audio.done` event

If raw chunked audio streaming is retained:

- include the same usage metadata in response headers when it is available before the response is finalized, or otherwise prefer SSE as the usage-reporting streaming mode

### Why this is preferable

- preserves the audio response body shape
- avoids adding a database or file-based logging subsystem
- keeps compatibility with clients expecting audio, not JSON wrappers
- is enough for proxy/accounting use cases where the caller just needs per-request usage data

## File-By-File Plan

## Phase 1: Stabilize The Minimal Core

### 1. Reduce `app/api/endpoints/speech.py`

Refactor this module into a smaller core built around:

- request validation
- model readiness check
- one internal generation function
- one or two response wrappers for streaming and non-streaming
- usage metadata calculation hooks

Remove from this file:

- upload endpoint support
- voice library resolution logic
- temp file upload handling
- memory cleanup counters and memory monitoring logs
- dependencies on `app/core/status.py`

Possible target shape:

- `generate_speech_internal(...)`
- `stream_speech_audio(...)`
- `stream_speech_sse(...)` if kept
- `text_to_speech(request: TTSRequest)`

### 2. Reduce `app/models/requests.py`

Keep only the fields that the minimal speech endpoint truly supports.

Remove:

- upload-related assumptions
- unused compatibility knobs
- any streaming fields that are not part of the final streaming design

### 3. Reduce `app/models/responses.py`

Keep only:

- `HealthResponse`
- `ErrorResponse`
- SSE response models if SSE is retained

Remove:

- `ConfigResponse`
- `ModelInfo`
- `ModelsResponse` if `/models` is removed
- `TTSProgressResponse`
- `TTSStatusResponse`
- `TTSStatisticsResponse`
- `APIInfoResponse`
- all voice library models

### 4. Reduce `app/models/__init__.py`

Update exports to match the reduced models only.

## Phase 2: Remove Bloat At The Router Level

### 5. Simplify `app/api/router.py`

Change router imports and includes to only:

- `speech`
- `health`


Everything else should be removed from the router before deleting the modules that implement it.

### 6. Simplify `app/main.py`

Change lifespan to:

- initialize model
- no voice library restoration
- no long-text background processor startup

Keep:

- FastAPI app creation
- CORS only if still required
- exception handlers

Consider simplifying further:

- if CORS is not needed for the reduced service, remove it
- if docs are not needed in production, leave them configurable rather than deleting outright

## Phase 3: Remove Entire Feature Subsystems

### 7. Delete voice library feature set

Delete:

- `app/core/voice_library.py`
- `app/api/endpoints/voices.py`
- any related helpers or response models
- tests dedicated to voice library behavior

Then clean up remaining references in:

- `app/main.py`
- `app/api/endpoints/speech.py`
- `app/models/__init__.py`
- docs and README

### 8. Delete long-text subsystem

Delete:

- `app/api/endpoints/long_text.py`
- `app/core/background_tasks.py`
- `app/core/long_text_jobs.py`
- `app/models/long_text.py`
- long-text-specific storage directories if they are part of the repo structure

Then clean up remaining references in:

- `app/main.py`
- `app/api/router.py`
- `app/models/__init__.py`
- `app/core/text_processing.py` if long-text-only functions remain there
- docs and README

### 9. Delete status and introspection subsystem

Delete:

- `app/core/status.py`
- `app/api/endpoints/status.py`
- `app/api/endpoints/config.py`
- `app/api/endpoints/memory.py`
- `app/api/endpoints/models.py` if not needed

Then clean up any imports and model definitions that existed only to support these routes.

## Phase 4: Remove Alias Infrastructure And Route Sprawl

### 10. Remove alias routing utilities

Delete or stop using:

- `app/core/aliases.py`

Replace generic aliasing with explicit route registration.

For the minimal API, use explicit decorators such as:

- `@router.post("/v1/audio/speech")`
- `@router.get("/health")`

Optional compatibility duplicates can still be explicit if needed, but the generic alias map is overkill for a two-endpoint service.

### 11. Standardize route naming

Recommendation:

- keep `/v1/audio/speech` as canonical
- optionally keep `/health`
- drop `/audio/speech`, `/tts`, `/v1/status`, `/routes`, and other aliases unless there is a proven consumer need

## Phase 5: Simplify Configuration

### 12. Reduce `app/config.py`

Keep only settings for:

- host and port
- model source
- model class
- model repo ID
- model revision
- model local path
- model cache dir
- Hugging Face token
- default language and supported languages if multilingual support remains
- voice sample path
- generation defaults: exaggeration, cfg_weight, temperature
- max input length
- no logging backend configuration should be introduced for this feature
- optional CORS setting if the service still needs it

Remove settings for:

- voice library directory
- long-text directories and limits
- memory cleanup intervals
- memory monitoring flags
- any retention settings for deleted subsystems

Also reduce validation logic accordingly.

### 13. Re-check `app/core/tts_model.py`

Keep this module largely intact because it already captures the useful model-loading behavior.

Potential cleanup:

- remove prints that mention deleted features such as voice library assumptions if any remain
- keep default, `hf_repo`, and `local_dir`
- keep multilingual support only if you still need it

## Phase 6: Add Usage Metadata To Responses

### 14. Compute `input_chars` and audio duration in the speech path

Implementation responsibilities:

- compute `input_chars` directly from the request
- compute audio duration from generated audio data rather than server generation time
- keep this logic close to the final response path so both streaming and non-streaming modes can expose the same metrics consistently

### 15. Wire usage metadata into the speech endpoint

For non-streaming:

- add `X-Usage-Input-Chars`
- add `X-Usage-Audio-Seconds`

For SSE streaming:

- include `input_chars`
- include audio duration in seconds
- place both in the final `speech.audio.done` event

If raw chunked audio streaming remains supported:

- either provide the same values in headers
- or document that SSE is the supported streaming mode for usage metadata

## Phase 7: Tests And Verification

### 16. Remove tests for deleted features

Delete or rewrite tests covering:

- voice library endpoints
- long text jobs
- status and statistics endpoints
- memory/config/info endpoints
- frontend behavior

Examples likely to remove or replace:

- `tests/test_voice_library.py`
- any long-text-specific test files
- any status/config/memory test files

### 17. Keep and strengthen minimal tests

Required tests:

- health endpoint responds during or after startup
- `/v1/audio/speech` returns success for valid input
- input validation rejects empty input
- input length enforcement works
- non-streaming returns audio bytes
- streaming returns the expected media type and event/audio shape
- non-streaming responses include `X-Usage-Input-Chars`
- non-streaming responses include `X-Usage-Audio-Seconds`
- SSE completion events include `input_chars` and audio duration in seconds
- HF repo and local model config validation still behaves correctly

### 18. Add regression checks for startup

Verify:

- app starts without voice library initialization
- app starts without long-text worker initialization
- model initialization state still reports correctly in health

## Phase 8: Docs, Packaging, And Repo Cleanup

### 19. Remove frontend

Delete:

- `frontend/`

Then remove any documentation or packaging references to it.

### 20. Simplify docs

Update or remove:

- `README.md`
- feature docs describing deleted endpoints
- examples for upload, voice library, long text jobs, status endpoints, and memory endpoints

The README should focus on:

- installation
- required environment variables
- startup
- one non-streaming curl example
- one streaming curl example
- model source configuration examples for default, HF repo, and local dir

### 21. Simplify Docker support

Current Docker setup appears broader than the minimal service needs.

Recommendation:

- keep one CPU image and one GPU image at most
- remove Docker variants that only existed to support broader deployment scenarios unless actively used

### 22. Review dependencies

Remove packages that become unused after cleanup, especially those introduced only for:

- long-text workflow
- frontend
- voice upload handling beyond standard FastAPI support
- memory/status extras

Then update:

- `requirements.txt`
- `pyproject.toml`

## Recommended Execution Sequence

### Step 1

Add response usage metadata while the current speech path still works.

Why:

- lowest-risk addition
- keeps the final metadata requirement small and explicit
- avoids introducing a subsystem that later has to be removed

### Step 2

Refactor `app/api/endpoints/speech.py` to isolate the minimal generation path.

Deliverable:

- `/v1/audio/speech` works without depending on voice library, uploads, long-text, or status manager

### Step 3

Simplify `app/api/router.py` and `app/main.py` to stop mounting and starting removed systems.

Deliverable:

- app starts with only speech and health concerns

### Step 4

Delete voice library subsystem and its tests.

### Step 5

Delete long-text subsystem and its tests.

### Step 6

Delete status/config/memory/info endpoints and in-memory tracking.

### Step 7

Remove alias routing and replace with explicit routes.

### Step 8

Reduce config, docs, and dependencies.

### Step 9

Run the reduced test suite and perform manual endpoint verification.

## Risks And Mitigations

### Risk 1: Shared logic is trapped inside `speech.py`

The long-text backend imports `generate_speech_internal(...)` from `app/api/endpoints/speech.py`, so deleting long-text before isolating the minimal speech path can create breakage.

Mitigation:

- first refactor speech generation into a self-contained internal path
- then delete long-text callers

### Risk 2: Hidden references to voice library remain

Voice resolution is currently intertwined with speech handling and startup initialization.

Mitigation:

- remove router references first
- then delete startup references
- then remove speech helper functions that resolve voice names via the library

### Risk 3: Config validation becomes inconsistent

`app/config.py` currently validates settings for multiple deleted subsystems.

Mitigation:

- trim config in one pass after deleted features are fully disconnected
- keep tests around model config validation

### Risk 4: Streaming usage metadata is inconsistent

Streaming requests often do not know final usage values until synthesis has completed.

Mitigation:

- treat the generator itself as the source of final usage values
- emit final usage metadata only in the completion event for SSE
- prefer SSE over raw chunked audio if identical usage reporting is required for all streaming clients

## Acceptance Criteria

The cleanup is complete when all of the following are true:

1. The app exposes only the intended minimal endpoints.
2. `POST /v1/audio/speech` supports the chosen streaming and non-streaming modes.
3. The service still loads default, HF repo, and local-dir models.
4. No voice library, long-text worker, or status/history subsystem code remains in runtime paths.
5. Successful responses expose `input_chars` and audio duration in seconds.
6. The reduced test suite passes.
7. The README reflects the reduced product accurately.
8. Unused dependencies and deleted modules are removed from the repo.

## Suggested Final Minimal Structure

One possible endpoint/core layout after cleanup:

- `app/main.py`
- `app/api/router.py`
- `app/api/endpoints/speech.py`
- `app/api/endpoints/health.py`
- `app/core/tts_model.py`
- `app/config.py`
- `app/models/requests.py`
- `app/models/responses.py`
- `tests/test_api.py`
- `tests/test_streaming.py`
- `tests/test_usage_metadata.py`

## Nice-To-Have Follow-Up After Cleanup

After the bloat removal is done, consider a second pass that is purely internal cleanup:

- split synthesis logic from HTTP routing completely
- replace `print(...)` with structured logging
- normalize error payloads across all remaining endpoints
- standardize the exact response header names and SSE completion payload shape

That second pass should happen after feature deletion, not before.
