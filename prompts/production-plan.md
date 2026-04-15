# Chatterbox TTS API Minimal Production-Safe Plan

## Goal

Harden the current FastAPI service enough to run reliably behind a real reverse proxy with predictable overload behavior, bounded resource usage, and basic operational visibility.

This plan intentionally avoids a larger architecture rewrite. It keeps:

- one FastAPI app
- one bounded in-process pool of model instances on one GPU or device
- the existing `POST /v1/audio/speech` contract
- the existing health and models endpoints
- the current chunked generation approach

It adds only the smallest set of controls needed to make the service safer under concurrent traffic.

## Production-Safe Definition

For this repo, a minimal production-safe version means:

- the service does not allow unbounded concurrent inference
- the service fails fast when overloaded instead of hanging or exhausting memory
- readiness is distinct from liveness so traffic is only sent to ready instances
- requests have bounded execution time
- streaming and non-streaming requests clean up correctly on failure or disconnect
- operators can see request volume, latency, overloads, and failures
- deployment guidance prevents accidental worker-level model duplication beyond the intended pool size

## Constraints And New Assumptions

- one shared global model instance is stored in `app/core/tts_model.py`
- `model.generate(...)` is called through `run_in_executor(...)`, but there is no concurrency limit
- `model.generate(...)` is not thread-safe, so concurrent inference cannot safely share one model instance
- the target GPU has enough VRAM headroom that a small number of separate model instances may fit on one device
- non-streaming requests buffer all generated chunk audio before returning
- startup initializes the model in the background, but request handling only discovers readiness at request time
- there is no backpressure, queue limit, request timeout, or overload response path
- there is no production-grade metrics or structured request logging

## End State

After this work, the service should behave like this:

1. A request arrives.
2. If the model is not ready, the service returns a readiness failure immediately.
3. If all model instances are busy, the service returns an overload response immediately or after a small bounded wait.
4. If accepted, the request runs within a defined timeout.
5. If the client disconnects or the timeout is hit, generation stops as cleanly as possible and resources are released.
6. The service emits logs and metrics for success, failure, timeout, and overload.

## Phase 1: Replace The Shared Model With A Bounded Model Pool

### Objective

Allow safe parallel requests on one GPU while keeping RAM or VRAM usage predictable.

### Phase 1 Decisions

Use these decisions as the default implementation unless benchmarking proves a different shape is necessary:

- Lease one model instance per request, not per chunk.
- Keep the lease for the full request lifetime.
- Do not switch model instances mid-request.
- Keep chunk generation sequential within a request.
- Use a FIFO async lease queue so waiting requests are handled in arrival order.
- Load the full pool eagerly at startup.
- Treat the instance as ready only when the full configured pool is loaded successfully.
- If any model instance fails during startup, fail startup rather than serving at partial capacity in the first version.
- If a leased model instance fails during request handling, treat that as a pool-health event and reduce capacity or mark the process unready.
- Do not use thread-based parallel access to a single model instance.

### Changes

- Replace the single global model instance with a small pool of separately loaded model instances.
- Route each accepted request to exactly one leased model instance for the duration of generation.
- Default the pool size conservatively:
  - start with `2`
  - tune upward only after measurement
- Add configuration values such as:
  - `MODEL_INSTANCE_COUNT`
  - `MAX_QUEUE_WAIT_SECONDS`
- Decide on one overload behavior:
  - wait briefly (10-15 sec) for a free model lease, then fail with `503`

### Concrete Runtime Shape

The smallest workable shape is:

- one `ModelPool` owner in `app/core/tts_model.py` or an adjacent module
- one list of loaded model instances
- one `asyncio.Queue` containing available model-slot identifiers or wrapper objects
- one small lease API, for example:
  - `initialize_model_pool()`
  - `acquire_model(timeout_seconds)`
  - `release_model(model_handle)`
  - `get_pool_status()`

The lease object should contain at minimum:

- stable instance ID
- model reference
- device
- health state if the instance has been marked degraded

### Request Leasing Semantics

The request lifecycle should be:

1. Validate request input.
2. Check readiness.
3. Attempt to lease one model instance.
4. If no lease is available within `MAX_QUEUE_WAIT_SECONDS`, return overload.
5. Use that same model instance for every chunk in the request.
6. Release the lease in a `finally` block.

Apply the same semantics to:

- non-streaming WAV generation
- SSE streaming generation

This avoids three classes of problems:

- thread-unsafe concurrent access to one model instance
- out-of-order or inconsistent per-request generation
- pool leaks when a request fails midway through chunk generation

### Implementation Notes

- Keep the pooling abstraction narrow: lease model, run request, release model.
- Do not share one model instance across concurrent requests.
- Keep chunk generation sequential within a request.
- Apply the same leasing policy to both streaming and non-streaming paths.
- Use `try/finally` around every lease.
- Keep request handlers unaware of the full pool internals beyond acquire or release.
- Prefer instance IDs in logs over object reprs.
- Keep the first version simple. Do not build a background job queue yet.

### Queue And Overload Rules

Define queue behavior explicitly:

- waiting for capacity is optional and bounded by `MAX_QUEUE_WAIT_SECONDS`
- `0` means fail immediately if no model instance is free
- queue order should be FIFO
- request validation should happen before entering the wait path
- queue wait time should count toward total request latency
- if request timeout expires while waiting for a lease, return timeout or overload consistently based on the chosen API contract

For the first version, the simplest recommended policy is:

- validate request
- attempt lease
- if no model is available within `MAX_QUEUE_WAIT_SECONDS`, return `503`

### Startup And Readiness Rules

Startup behavior should be explicit:

- create the configured number of model instances eagerly during startup
- populate the lease queue only after each instance is fully loaded
- fail startup if any configured instance fails to load
- mark readiness true only after the full pool is loaded
- expose pool size and readiness state in health output

This keeps the first release operationally simple:

- no partial-capacity startup
- no lazy first-request loading
- no hidden warm-up path under live traffic

### Broken Instance Policy

If a leased model instance fails during request handling:

- log the instance ID and request ID
- mark the instance unhealthy
- do not immediately return it to the available queue
- reduce effective pool capacity
- if healthy capacity reaches `0`, mark the service unready

For the minimal version, do not implement live replacement inside the process. Recovery can be:

- process restart by the supervisor, or
- manual restart after investigation

This is intentionally conservative and avoids trying to hot-rebuild pool members inside a damaged process.

### Executor And Threading Rules

The core safety constraint is:

- one model instance must never run more than one active `generate(...)` call at a time

Phase 1 does not require a dedicated executor per instance, but it does require:

- no concurrent use of one model instance
- no logic that reacquires a different thread-unsafe instance during a request
- clear documentation that pool parallelism comes from multiple model instances, not threads on one instance

If the default executor is retained, the lease boundary is what provides safety.

### VRAM Sizing Rules

Do not size the pool from idle model memory alone.

Use this procedure:

1. Measure VRAM after one model instance loads and is idle.
2. Measure peak VRAM during one active request.
3. Estimate active overhead per additional model instance and per in-flight request.
4. Keep a safety margin for fragmentation and transient spikes.
5. Choose the largest pool size that stays comfortably below device limits under representative concurrent load.

For the first production-safe version:

- start with `MODEL_INSTANCE_COUNT=2`
- benchmark `2`, then `3`, then `4`
- stop increasing when p95 latency, instability, or VRAM pressure becomes unacceptable

### Health And Status Exposure

Phase 1 should expose enough pool status to operate the service safely.

At minimum, `/health` should report:

- configured pool size
- ready instance count
- available instance count
- busy instance count
- degraded or unhealthy instance count
- readiness boolean derived from usable pool state

The first version does not need historical pool metrics, but it does need current pool state.

### Minimum Logging Requirements For Phase 1

Even before the broader observability phase, Phase 1 should log:

- request ID
- leased model instance ID
- lease wait time
- request mode: streaming or non-streaming
- success, overload, timeout, or generation failure

Do not log full request text.

### Minimum Test Coverage For Phase 1

Before moving to later phases, prove that the pool works correctly.

Required tests:

- two concurrent requests succeed when `MODEL_INSTANCE_COUNT=2`
- a third concurrent request waits or fails according to configured policy
- a failed request returns its lease if the model instance remains healthy
- a broken model instance is removed from availability
- readiness becomes false when no healthy instances remain
- SSE and non-streaming both use one stable instance for the entire request

### Acceptance Criteria

- concurrent requests use at most the configured number of model instances
- overload behavior is deterministic and documented
- both SSE and non-streaming follow the same model-leasing policy
- every accepted request uses exactly one stable leased model instance
- leases are always returned or retired correctly after success, timeout, disconnect, or failure
- readiness reflects real usable pool capacity

## Phase 2: Add Readiness And Overload Semantics

### Objective

Make it safe to place the service behind a load balancer or container orchestrator.

### Changes

- Split health behavior into two meanings:
  - liveness: process is up
  - readiness: the full configured model pool is loaded and able to accept work
- Keep `/health` if desired, but ensure it clearly exposes readiness state.
- Optionally add a dedicated readiness endpoint if the deployment platform expects one.
- Return explicit overload errors when all model instances are busy.
- Document expected status codes for:
  - model not ready
  - overload
  - invalid request
  - internal failure

### Implementation Notes

- Replace or extend the existing model initialization state in `app/core/tts_model.py` to account for pool initialization.
- Avoid sending traffic to instances where initialization is still in progress.
- Prefer clear machine-readable error payloads that match the current error response shape.

### Acceptance Criteria

- an instance is only considered ready when the configured model pool is actually usable
- overloads can be distinguished from ordinary internal errors
- deployment docs can point traffic only to ready instances

## Phase 3: Add Request Timeouts And Cancellation Handling

### Objective

Prevent requests from running forever and reduce resource leaks when clients disappear while models are leased.

### Changes

- Add a configurable total request timeout, such as `REQUEST_TIMEOUT_SECONDS`.
- Wrap generation in timeout handling for both streaming and non-streaming paths.
- Detect client disconnects during SSE streaming and stop generating remaining chunks.
- Ensure partial buffers and tensors are released on timeout or disconnect.
- Ensure leased model instances are always returned to the pool.

### Implementation Notes

- Timeouts should be applied around the full generation lifecycle, not just network writes.
- Keep the behavior minimal and explicit:
  - if timed out, return or terminate with a clear timeout error
  - do not retry automatically
- If some lower-level inference work cannot be interrupted immediately, document that limitation and still stop any further chunk scheduling.
- Use `try/finally` semantics around model leasing so pool capacity is not lost after failures.

### Acceptance Criteria

- long-running requests terminate within a predictable time bound
- SSE clients that disconnect do not continue consuming the full remaining request budget
- failures leave the process and the model pool in a recoverable state for later requests

## Phase 4: Reduce Memory Risk

### Objective

Keep one request from making concurrent traffic unstable.

### Changes

- Keep `MAX_TOTAL_LENGTH` strict and conservative.
- Consider using a stricter effective limit for non-streaming than for SSE if needed after testing.
- Add a soft memory safety check before accepting new inference work if process memory is already above a threshold.
- Size the model pool conservatively against real VRAM use, not just idle model footprint.
- Prefer SSE for larger accepted requests in documentation and operational guidance.

### Implementation Notes

- The current non-streaming path holds all chunk audio before writing the WAV, so it remains the riskiest path.
- The total VRAM budget must include model weights plus per-request generation overhead for each pooled instance.
- The minimal version does not need a full memory manager. A simple pre-admission guard is enough if it proves necessary.
- Keep the first release conservative rather than trying to maximize throughput.

### Acceptance Criteria

- pooled concurrent medium-to-large requests do not trigger obvious memory blowups under tested limits
- operators have a documented safe input-size and concurrency envelope

## Phase 5: Add Basic Observability

### Objective

Make overload, latency, and failure modes visible.

### Changes

- Add structured logs for each request.
- Include a request identifier in logs and response headers if practical.
- Record at minimum:
  - route
  - request ID
  - input character count
  - streaming vs non-streaming mode
  - chunk count
  - queue or lease wait time if any
  - leased model instance identifier if practical
  - total latency
  - success or failure outcome
  - timeout and overload counts
- Add a minimal metrics endpoint or metrics integration if the deployment stack already expects Prometheus-style scraping.

### Implementation Notes

- Keep logs concise and machine-friendly.
- Do not log full input text.
- If metrics are too much for the first pass, structured logs are the minimum acceptable starting point.

### Acceptance Criteria

- operators can tell whether the service is slow, overloaded, timing out, or failing
- logs are usable without exposing request text

## Phase 6: Deployment Guardrails

### Objective

Prevent unsafe production deployment defaults.

### Changes

- Document the supported deployment shape clearly:
  - one app worker per process
  - one bounded model pool per process
  - no blind scaling of worker count inside a single container
- Document reverse proxy expectations:
  - body size limit
  - request timeout
  - idle timeout for SSE
- Document recommended environment defaults for production:
  - `MODEL_INSTANCE_COUNT=2` as a starting point on large-memory GPUs
  - conservative request timeout
  - conservative text length limit
- Add startup validation for any new production-safety config values.

### Implementation Notes

- The main operational footgun is accidentally multiplying the intended model pool by running multiple server workers.
- The docs should make that unsafe by default configuration obvious.

### Acceptance Criteria

- deployment docs reduce the chance of accidental pool multiplication and runaway memory usage
- production defaults favor safety over peak throughput

## Phase 7: Validation And Load Testing

### Objective

Verify the chosen safety limits with real behavior instead of assumptions.

### Changes

- Add tests for:
  - not-ready requests
  - overload rejection
  - timeout behavior
  - SSE disconnect handling where practical
- Run manual or scripted concurrency checks at small scale:
  - `2` concurrent requests
  - `5` concurrent requests
  - `10` concurrent requests
- Measure:
  - latency
  - memory growth
  - VRAM growth
  - per-instance utilization if visible
  - overload rate
  - recovery after failures

### Implementation Notes

- The goal is not benchmarking maximum throughput yet.
- The goal is to prove the service stays stable when pushed past its configured safe envelope.

### Acceptance Criteria

- the documented pool size and timeout values are based on observed behavior
- the service recovers cleanly after overloads and timeouts

## Recommended Minimal Sequence

Implement the work in this order:

1. Replace the shared model with a bounded model pool and overload response.
2. Expose clear readiness state and deployment guidance.
3. Add request timeout handling.
4. Add structured request logging.
5. Add overload and timeout tests.
6. Run small-scale concurrency validation and tune pool size.

## Explicit Non-Goals For The Minimal Version

These can wait until later:

- a distributed job queue
- persistent request history
- autoscaling logic based on metrics
- multi-model routing
- parallel chunk generation within one request
- aggressive throughput optimization

## Proposed Config Additions

Suggested new environment variables:

- `MODEL_INSTANCE_COUNT=2`
- `MAX_QUEUE_WAIT_SECONDS=0`
- `REQUEST_TIMEOUT_SECONDS=120`
- `MEMORY_SOFT_LIMIT_MB=` optional
- `ENABLE_REQUEST_LOGGING=true`

The effective parallelism should equal the model pool size. Do not add thread-based parallel access to a single model instance.

Keep them validated in `app/config.py` and expose the non-sensitive values in `/health` if useful.

## Checklist

- [ ] Replace the single shared model with a bounded pool of model instances
- [ ] Add configurable model-pool size and queue-wait settings
- [ ] Return explicit overload errors when all model instances are busy
- [ ] Make readiness clearly distinguishable from liveness
- [ ] Ensure deployment only routes traffic to ready instances
- [ ] Add a configurable total request timeout
- [ ] Stop or curtail work on SSE client disconnects
- [ ] Ensure failures and timeouts release request-scoped resources and return leased models to the pool
- [ ] Keep input-size limits conservative and documented
- [ ] Add structured request logging without logging full input text
- [ ] Measure VRAM usage as pool size increases
- [ ] Add overload and timeout test coverage
- [ ] Run small concurrency validation and tune the safe pool size
- [ ] Document production deployment constraints and recommended env values
