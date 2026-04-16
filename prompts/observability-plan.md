# Chatterbox TTS API Observability Plan

## Goal

Add practical observability to the current service so operators can answer four questions quickly:

1. Is the service healthy and receiving traffic?
2. How many requests succeed, fail, time out, or overload?
3. How long do requests spend waiting, generating, and finishing?
4. Is the model pool being saturated?

This plan focuses on two concrete deliverables:

- structured logs
- Prometheus metrics on `GET /metrics`

## Current State

The current service can run without additional production-plan phases. The remaining non-observability items are operational improvements, not blockers for basic execution.

What already exists:

- request IDs in speech responses
- some request-scoped logging in `app/api/endpoints/speech.py`
- pool status in `/health`
- explicit readiness, capacity, and timeout behavior

What is missing:

- consistent structured log format across the app
- request lifecycle logs for all request outcomes
- a scrapeable metrics endpoint
- counters and histograms for request behavior and pool pressure

## Current Status

Completed in the current branch:

- shared JSON structured logging utility
- structured startup and shutdown logs
- structured model-pool initialization and retirement logs
- structured request logs for start, lease acquisition, success, timeout, overload, disconnect, and failure
- focused unit coverage for structured log payloads and field safety
- Prometheus metrics module under `app/core/metrics.py`
- `GET /metrics` endpoint with Prometheus text exposition output
- request counters, latency histograms, usage histograms, pool gauges, and pool event counters
- focused unit coverage for metrics rendering and key request outcomes

The next observability step is live scrape verification and any metric tuning based on real traffic.

## Desired End State

After this work:

- every important request outcome emits a structured log entry
- logs include enough fields to correlate request timing, route, mode, and model-instance usage
- `GET /metrics` exposes Prometheus-compatible metrics
- operators can see request volume, latency, overloads, timeouts, disconnects, and pool utilization
- observability additions do not change the API contract for normal callers

## Design Principles

1. Prefer a small number of high-value metrics over a large unfocused set.
2. Keep log fields stable and machine-friendly.
3. Do not log input text or other sensitive request payloads.
4. Put shared observability code in one small module rather than scattering it across routes.
5. Instrument the existing request lifecycle rather than redesigning it.

## Phase 1: Structured Logging (Completed)

### Objective

Make request and runtime behavior inspectable through consistent structured logs.

### Decisions

- Use JSON logs as the default format.
- Log one event per important request lifecycle milestone rather than every tiny step.
- Keep the current request ID and use it as the primary correlation field.
- Include model instance ID when a lease has been acquired.
- Do not log request input text.
- Do not add a third-party logging stack unless the standard library becomes clearly insufficient.

### Scope

Add a small logging utility module, for example `app/core/observability.py`, that provides:

- logger configuration
- JSON log formatting
- helpers for request-scoped logging fields

Update the runtime to log at minimum:

- app startup begun
- model pool initialization begun
- model pool initialization succeeded
- model pool initialization failed
- request accepted into runtime processing
- model lease acquired
- request completed successfully
- request timed out
- request rejected for no capacity
- SSE client disconnected
- request failed with generation or internal error
- model instance retired from the pool

### Required Log Fields

Every request-scoped log should include at minimum:

- `event`
- `request_id`
- `route`
- `request_mode`
- `status` or `outcome`
- `elapsed_seconds`

When available, also include:

- `model_instance_id`
- `input_chars`
- `audio_seconds`
- `lease_wait_seconds`
- `chunk_count`
- `error_type`
- `timeout_stage`

App-level logs should include at minimum:

- `event`
- `service`
- `version`
- `device`
- `configured_pool_size`

### Implementation Notes

- The current `_log_request_event(...)` helper in `app/api/endpoints/speech.py` should be replaced or routed through the shared observability module.
- Keep one log emission for success and one for failure, rather than logging every chunk.
- Emit success logs after the outcome is known:
  - non-streaming: after response generation completes
  - SSE: after `speech.audio.done` is emitted
- Emit timeout and disconnect logs from the existing Phase 2 control points.
- Keep startup logging in `app/main.py` and `app/core/tts_model.py` consistent with the same structured format.

### Acceptance Criteria

- logs are valid JSON or another explicitly chosen structured format
- request logs are consistent across success, timeout, overload, disconnect, and failure
- operators can correlate logs using `request_id`
- logs do not include full input text

### Completed Notes

The current implementation includes these Phase 1 behaviors:

- root logging configured with a shared JSON formatter
- shared `log_event(...)` helper for stable structured fields
- startup and shutdown logs in `app/main.py`
- model-pool lifecycle logs in `app/core/tts_model.py`
- request outcome logs in `app/api/endpoints/speech.py`
- request ID, route, mode, elapsed time, and outcome fields on request-scoped logs
- unit coverage for formatter output and request-log field presence

## Phase 2: Prometheus Metrics Endpoint (Completed)

### Objective

Expose scrapeable metrics for traffic, latency, failures, and model-pool behavior.

### Decisions

- Use the Prometheus Python client library.
- Expose metrics on `GET /metrics`.
- Return standard Prometheus text exposition format.
- Keep metrics process-local; do not add aggregation logic in-app.
- Instrument only the active API surface.

### Scope

Add a small metrics module, for example `app/core/metrics.py`, that defines and exports:

- counters
- histograms
- gauges
- helper functions for updating metrics from request handlers and model-pool events

Add a new endpoint module, for example `app/api/endpoints/metrics.py`, that returns the registry contents.

Mount that route from `app/api/router.py`.

### Recommended Metrics

Request counters:

- total requests by route and mode
- successful requests by route and mode
- failed requests by route and mode
- timed-out requests by route and mode
- overloaded requests by route and mode
- SSE disconnects by route

Latency histograms:

- total request duration seconds
- model lease wait seconds
- audio generation duration seconds

Usage histograms:

- input characters per request
- audio seconds per successful request
- chunk count per request

Pool gauges:

- configured model instances
- healthy model instances
- available model instances
- busy model instances
- unhealthy model instances

Pool event counters:

- model instance retirements
- lease acquisition failures due to no capacity

### Label Strategy

Keep labels small and bounded.

Recommended labels:

- `route`
- `mode` with values like `audio` and `sse`
- `outcome` only where cardinality remains low

Avoid labels with unbounded values such as:

- `request_id`
- raw exception messages
- model instance ID on metrics

### Implementation Notes

- Update request metrics from the same request lifecycle points used for structured logs.
- Record pool gauges from the pool status helpers rather than duplicating state.
- For latency histograms, prefer observing once per completed request rather than once per chunk.
- For SSE disconnects, count the disconnect event even though the request has no final completion event.
- Keep `/metrics` outside the normal TTS request path and do not require a model lease for it.

### Acceptance Criteria

- `/metrics` returns valid Prometheus output
- metrics include request counts, latency, and pool status
- labels remain bounded and operationally safe
- metrics can distinguish success, timeout, overload, and disconnect behavior

### Completed Notes

The current implementation includes these Phase 2 behaviors:

- Prometheus client dependency in `pyproject.toml`
- shared metrics module in `app/core/metrics.py`
- `GET /metrics` endpoint in `app/api/endpoints/metrics.py`
- mounted metrics route in `app/api/router.py`
- request counters by route, mode, and outcome
- latency histograms for total request time, lease wait time, and generation time
- usage histograms for input chars, audio seconds, and chunk count
- pool gauges for configured, healthy, available, busy, and unhealthy instances
- pool event counters for lease acquisition failures and model instance retirements
- unit coverage for metrics rendering, success, overload, and SSE disconnect updates

## Phase 3: Integration Points

### Objective

Wire observability into the app with minimal intrusion.

### Changes

- add shared observability helpers under `app/core`
- update `app/main.py` startup logging
- update `app/core/tts_model.py` to emit pool lifecycle metrics and logs
- update `app/api/endpoints/speech.py` to emit structured request logs and metrics
- add `app/api/endpoints/metrics.py`
- mount `/metrics` in `app/api/router.py`
- add any required dependency to `pyproject.toml`

### Acceptance Criteria

- observability code stays centralized and small
- request handlers do not become substantially more complex than they are now
- `/metrics` does not interfere with existing API endpoints

## Phase 4: Verification

### Objective

Prove the new observability behavior is correct and stable.

### Tests

Add tests where practical for:

- `/metrics` endpoint responds successfully
- metrics output contains expected counters and gauges
- a successful request updates success metrics
- a timeout updates timeout metrics
- an overload updates overload metrics
- SSE disconnect updates disconnect metrics

For structured logging, use focused unit tests where practical to verify:

- required fields are present
- logs do not contain raw input text
- timeout and disconnect logs include the correct outcome fields

### Manual Verification

Run a short manual check that confirms:

- `/metrics` is scrapeable while the app is serving requests
- logs are readable and structured in the target runtime
- counters increase as expected under success, overload, and timeout scenarios

## Open Questions

These do not block the initial implementation, but they should be kept in mind:

1. Do you want pure JSON logs always, or an env switch for human-readable local logs?
2. Do you want `/metrics` always enabled, or gated by config?
3. Do you want startup and health metrics beyond the request and pool metrics listed here?

If not otherwise specified, the default recommendation is:

- JSON logs always
- `/metrics` enabled by default
- keep the first metrics set focused on requests and pool status

## Checklist

- [x] Add a shared structured logging utility
- [x] Emit structured startup and model-pool lifecycle logs
- [x] Emit structured request outcome logs for success, timeout, overload, disconnect, and failure
- [x] Avoid logging full request text
- [x] Add Prometheus client dependency if needed
- [x] Add a `/metrics` endpoint
- [x] Add request counters for success, timeout, overload, failure, and disconnect
- [x] Add latency histograms for total request time and lease wait time
- [x] Add gauges for current model-pool state
- [x] Add tests for `/metrics` output and key metric updates
- [x] Add focused tests for structured log field presence where practical
- [ ] Run manual verification against a live server
