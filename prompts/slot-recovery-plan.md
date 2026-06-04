# Slot Recovery & Periodic Refresh Plan

## Problems

### 1. Permanent slot failure

Once a slot is retired via `_update_runtime_after_slot_failure` it is gone forever.
If all slots fail the pool enters `ERROR` state and the service returns 503 on every
subsequent request until the pod is manually restarted.

### 2. Memory and state accumulation

`AlignmentStreamAnalyzer.__init__` (chatterbox library) calls
`register_forward_hook` on transformer attention layers on every inference call.
The returned handles are never explicitly removed. Over many requests these hooks
accumulate on the model, causing progressively slower generation and growing GPU
memory consumption. A full model reload is the only reliable way to clear them. This
is a known pattern for long-running PyTorch inference services.

---

## Solution: Shared reinit core with two triggers

Both problems are solved by the same operation — unload the old model instance and
load a fresh one — so they share a single `_reinitialize_slot` coroutine with two
call sites:

| Trigger | When | Backoff |
|---|---|---|
| **Failure recovery** | Slot retired by `_update_runtime_after_slot_failure` | Exponential, up to `MAX_SLOT_RECOVERY_ATTEMPTS` |
| **Scheduled refresh** | Slot has served `SLOT_REFRESH_AFTER_REQUESTS` successful requests | None — slot is healthy, reload immediately |

### Rolling constraint

With `MODEL_INSTANCE_COUNT=2` on a single GPU, only one slot should ever be
reinitializing at a time — guaranteeing at least one slot always serves requests. A
module-level `asyncio.Lock` (`_reinit_lock`) enforces this:

- **Recovery**: always schedules a task regardless of lock state. The task waits
  for the lock before proceeding. The failing slot is already out of the pool, so
  waiting is safe.
- **Refresh**: only starts if `not _reinit_lock.locked()`. If the lock is busy
  the slot returns to the pool and re-checks on the next request completion. The
  refresh triggers as soon as the previous reinit finishes.

### Single-GPU unload constraint

Loading a new model instance before unloading the old one would OOM on a
single-GPU deployment. `_reinitialize_slot` therefore sets `slot.model = None`
and calls `torch.cuda.empty_cache()` before attempting the load.

---

## Implementation

### 1. Add module-level globals

```python
_reinit_lock: Optional[asyncio.Lock] = None
```

Initialise it inside `initialize_model` alongside `_available_model_ids`:

```python
_reinit_lock = asyncio.Lock()
```

Also reset it in `_reset_runtime_state`:

```python
_reinit_lock = None
```

### 2. Add constants

```python
# Reinitialize a failed slot up to this many times before giving up permanently.
MAX_SLOT_RECOVERY_ATTEMPTS: int = 3

# Initial backoff for failure recovery; doubles on each retry (5s → 10s → 20s).
SLOT_RECOVERY_INITIAL_BACKOFF_SECONDS: float = 5.0

# Proactively reload a slot after this many requests to clear accumulated
# attention hooks and GPU memory fragmentation.
SLOT_REFRESH_AFTER_REQUESTS: int = 200
```

### 3. Extend `ModelSlot`

```python
@dataclass
class ModelSlot:
    instance_id: int
    model: Any
    device: str
    healthy: bool = True
    last_error: Optional[str] = None
    consecutive_failures: int = 0      # existing (exception-handling plan)
    requests_served: int = 0           # NEW: counts successful completions
    reinitializing: bool = False       # NEW: True while a reinit task is running
```

### 4. Add `_reinitialize_slot` coroutine

Place this after `_update_runtime_after_slot_failure`. It handles both failure
recovery and scheduled refresh.

```python
async def _reinitialize_slot(instance_id: int, reason: str) -> None:
    global _initialization_state, _reinit_lock

    if _reinit_lock is None:
        return

    async with _reinit_lock:
        slot = _model_pool[instance_id]
        slot.reinitializing = True
        loop = asyncio.get_running_loop()

        log_event(
            logger, logging.INFO, "model_instance_reinit_started",
            model_instance_id=instance_id, reason=reason,
        )

        # Free GPU memory before loading. On a single-GPU deployment both
        # instances share the same device; the old model must be released first.
        slot.model = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        is_recovery = reason == "failure"
        max_attempts = MAX_SLOT_RECOVERY_ATTEMPTS if is_recovery else 2
        backoff = SLOT_RECOVERY_INITIAL_BACKOFF_SECONDS if is_recovery else 0.0

        for attempt in range(1, max_attempts + 1):
            if backoff > 0:
                await asyncio.sleep(backoff)
                backoff *= 2

            log_event(
                logger, logging.INFO, "model_instance_reinit_attempt",
                model_instance_id=instance_id, reason=reason, attempt=attempt,
            )

            try:
                model_source = Config.get_model_source()
                model_class = Config.get_model_class()
                new_model, _ = await loop.run_in_executor(
                    None,
                    lambda ms=model_source, mc=model_class, dv=slot.device: (
                        _load_model_sync(ms, mc, dv)
                    ),
                )
            except Exception as exc:
                log_event(
                    logger, logging.WARNING, "model_instance_reinit_attempt_failed",
                    model_instance_id=instance_id, reason=reason,
                    attempt=attempt, error=str(exc),
                )
                if attempt == max_attempts:
                    slot.healthy = False
                    slot.reinitializing = False
                    _update_runtime_after_slot_failure(
                        instance_id, f"reinit exhausted after {attempt} attempts: {exc}"
                    )
                    return
                continue

            # Success
            slot.model = new_model
            slot.healthy = True
            slot.last_error = None
            slot.requests_served = 0
            slot.consecutive_failures = 0
            slot.reinitializing = False

            if _available_model_ids is not None:
                _available_model_ids.put_nowait(instance_id)

            if _initialization_state == InitializationState.ERROR.value:
                _initialization_state = InitializationState.READY.value
                _initialization_progress = (
                    f"Pool recovered: {_healthy_slot_count()}/{len(_model_pool)} instances healthy"
                )

            observe_pool_status(get_pool_status())
            log_event(
                logger, logging.INFO, "model_instance_reinit_completed",
                model_instance_id=instance_id, reason=reason,
                healthy_instances=_healthy_slot_count(),
            )
            return
```

### 5. Trigger recovery from `_update_runtime_after_slot_failure`

Add at the end of the existing function body:

```python
    # Schedule background recovery. The task waits for _reinit_lock, so it
    # won't run concurrently with another reinit even if scheduled immediately.
    slot = _model_pool[instance_id]
    if not slot.reinitializing:
        asyncio.create_task(_reinitialize_slot(instance_id, "failure"))
```

### 6. Trigger scheduled refresh from `release_model_lease`

In the successful-release path (after the `soft_failure` / `consecutive_failures`
block, before `put_nowait`):

```python
    slot.requests_served += 1

    if (
        slot.requests_served >= SLOT_REFRESH_AFTER_REQUESTS
        and _reinit_lock is not None
        and not _reinit_lock.locked()
    ):
        # Lock is free — take this slot out for a proactive refresh now.
        # The slot is NOT returned to the pool here; _reinitialize_slot does that.
        slot.reinitializing = True
        asyncio.create_task(_reinitialize_slot(slot.instance_id, "scheduled_refresh"))
        observe_pool_status(get_pool_status())
        return
    # Lock is busy — put the slot back; it will re-check next request completion.
```

`requests_served` is not reset when the lock is busy, so the check fires again on
the next request until the lock is free.

### 7. Update the liveness probe (`app/api/endpoints/health.py`)

Extend `liveness_probe` to stay alive while a reinit task is in progress:

```python
async def liveness_probe(response: Response):
    from app.core import tts_model as _tts_model
    init_state = get_initialization_state()
    if init_state == InitializationState.ERROR.value:
        recovering = any(s.reinitializing for s in _tts_model._model_pool)
        if not recovering:
            response.status_code = 503
            return {"status": "dead", "reason": "model pool permanently failed"}
    return {"status": "alive"}
```

---

## Behaviour summary

| Scenario | Before | After |
|---|---|---|
| Single slot fails | Slot retired permanently | Recovery task scheduled; slot reloads with backoff |
| All slots fail | Pod dead until restart | Recovery tasks run; pool resets to READY on first success |
| Slot serves 200 requests | Hook accumulation, no action | Slot taken offline, reloaded, returned to pool |
| Refresh attempted while recovery running | N/A | Refresh deferred; slot keeps serving until lock is free |
| Reload itself fails | N/A | Retries (2× for refresh, 3× for recovery); marks unhealthy only on exhaustion |

---

## Summary of file changes

| File | Change |
|---|---|
| `app/core/tts_model.py` | Add `_reinit_lock` global; initialise in `initialize_model`, reset in `_reset_runtime_state` |
| `app/core/tts_model.py` | Add `MAX_SLOT_RECOVERY_ATTEMPTS`, `SLOT_RECOVERY_INITIAL_BACKOFF_SECONDS`, `SLOT_REFRESH_AFTER_REQUESTS` constants |
| `app/core/tts_model.py` | Add `requests_served`, `reinitializing` fields to `ModelSlot` |
| `app/core/tts_model.py` | Add `_reinitialize_slot` coroutine |
| `app/core/tts_model.py` | Trigger recovery at end of `_update_runtime_after_slot_failure` |
| `app/core/tts_model.py` | Trigger refresh in `release_model_lease` after successful request |
| `app/api/endpoints/health.py` | Extend `liveness_probe` to stay alive during active reinit |
