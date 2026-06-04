# Queue Exhaustion Plan: Fast-Fail When Pool Enters ERROR State

## Problem

When a slot failure causes the pool to enter `ERROR` state, requests that are
*already waiting* inside `acquire_model_lease` do not find out immediately. They
are blocked on:

```python
instance_id = await asyncio.wait_for(queue.get(), timeout=wait_seconds)
```

Because the broken slot's ID was never returned to `_available_model_ids`, the queue
is now permanently empty. The waiting requests burn their full `MAX_QUEUE_WAIT_SECONDS`
timeout (default: `Config.MAX_QUEUE_WAIT_SECONDS`) before giving up with
`ModelPoolExhaustedError`. Under load this means every in-flight request holds a
connection and an asyncio task open for the entire timeout window, potentially
exhausting the server's connection pool and delaying the 503 responses that callers
need to trigger their own retry logic.

There is also a secondary issue in the dequeue loop (`tts_model.py:439-440`): when
an unhealthy slot ID is encountered it is silently dropped with `continue` and the
timeout countdown is not reset. If multiple stale IDs pile up the caller may wait
much longer than expected before receiving `ModelPoolExhaustedError`.

---

## Solution: Pool state change event

Add an `asyncio.Event` (`_pool_error_event`) that is set whenever the pool enters
`ERROR` state. `acquire_model_lease` waits on *both* the queue and the event
simultaneously; whichever fires first wins. When the event fires the function raises
`ModelNotReadyError` immediately rather than waiting for the timeout.

---

## Implementation

### 1. Add `_pool_error_event` to module globals (`app/core/tts_model.py`)

```python
_pool_error_event: Optional[asyncio.Event] = None
```

Initialise it during `initialize_model` (alongside `_available_model_ids`):

```python
# near line 310, after `available_ids: asyncio.Queue[int] = asyncio.Queue()`
_pool_error_event = asyncio.Event()
```

Also reset it in `_reset_runtime_state`:

```python
global ..., _pool_error_event
...
_pool_error_event = None
```

### 2. Set the event in `_update_runtime_after_slot_failure` (`tts_model.py:230`)

```python
if healthy_count <= 0:
    _initialization_state = InitializationState.ERROR.value
    _initialization_progress = "No healthy model instances available"
    if _pool_error_event is not None:
        _pool_error_event.set()     # NEW: wake all waiting acquirers
```

If the slot recovery plan is also implemented, clear the event when a slot
successfully recovers:

```python
# in _recover_slot, after restoring the slot
if _pool_error_event is not None:
    _pool_error_event.clear()
```

### 3. Update `acquire_model_lease` (`tts_model.py:412`)

Replace the current `while True` loop body with a dual-wait pattern:

```python
async def acquire_model_lease(timeout_seconds: Optional[float] = None) -> ModelLease:
    if not is_ready() or _available_model_ids is None:
        raise ModelNotReadyError("Model pool not ready")

    wait_seconds = (
        Config.MAX_QUEUE_WAIT_SECONDS if timeout_seconds is None else timeout_seconds
    )
    queue = _available_model_ids
    error_event = _pool_error_event

    while True:
        # Fast-fail if the pool has already entered ERROR state
        if error_event is not None and error_event.is_set():
            raise ModelNotReadyError("Model pool entered error state")

        try:
            if wait_seconds <= 0:
                instance_id = queue.get_nowait()
            else:
                # Race the queue against the pool error event so we don't wait
                # the full timeout when the pool is permanently down.
                queue_task = asyncio.ensure_future(queue.get())
                if error_event is not None:
                    event_task = asyncio.ensure_future(error_event.wait())
                    done, pending = await asyncio.wait(
                        {queue_task, event_task},
                        timeout=wait_seconds,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for t in pending:
                        t.cancel()
                    if not done:
                        queue_task.cancel()
                        raise asyncio.TimeoutError
                    if event_task in done:
                        raise ModelNotReadyError("Model pool entered error state")
                    instance_id = queue_task.result()
                else:
                    instance_id = await asyncio.wait_for(queue.get(), timeout=wait_seconds)
        except asyncio.QueueEmpty as exc:
            raise ModelPoolExhaustedError("No model instances available") from exc
        except asyncio.TimeoutError as exc:
            raise ModelPoolExhaustedError(
                "Timed out waiting for an available model instance"
            ) from exc

        if instance_id >= len(_model_pool):
            continue

        slot = _model_pool[instance_id]
        if not slot.healthy:
            continue

        lease = ModelLease(
            instance_id=slot.instance_id,
            model=slot.model,
            device=slot.device,
        )
        observe_pool_status(get_pool_status())
        return lease
```

The `asyncio.ensure_future` / `asyncio.wait` pattern is the standard way to race
two awaitables in asyncio. The `event_task` future is cancelled as soon as the
queue delivers an ID, so the event listener does not accumulate.

### 4. Handle `ModelNotReadyError` from within the wait (`app/api/endpoints/speech.py`)

`_acquire_request_lease` already catches `ModelNotReadyError` and returns 503, so
no changes are needed in the speech endpoint. The new fast-fail path surfaces via
the same exception type.

---

## Behaviour change summary

| Scenario | Before | After |
|---|---|---|
| Pool enters ERROR while request is queued | Wait `MAX_QUEUE_WAIT_SECONDS`, then 503 | Immediate 503 |
| Pool enters ERROR before request arrives | Immediate 503 (unchanged) | Immediate 503 |
| Normal request, slot available | Returns immediately (unchanged) | Returns immediately |
| Normal request, slot busy | Waits up to timeout (unchanged) | Waits up to timeout |

---

## Summary of file changes

| File | Change |
|---|---|
| `app/core/tts_model.py` | Add `_pool_error_event: Optional[asyncio.Event]` global |
| `app/core/tts_model.py` | Initialise and reset event in `initialize_model` / `_reset_runtime_state` |
| `app/core/tts_model.py` | Set event in `_update_runtime_after_slot_failure` when healthy count hits 0 |
| `app/core/tts_model.py` | Update `acquire_model_lease` to race queue against error event |
