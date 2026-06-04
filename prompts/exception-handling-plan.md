# Exception Handling Plan: Preventing Cascading Model Slot Failures

## Problem

Any exception thrown by `lease.model.generate()` inside `_generate_chunk_audio`
(`app/api/endpoints/speech.py:354`) currently calls `lease.mark_broken()`, which
permanently retires the model slot. When all slots are gone,
`_initialization_state` flips to `ERROR` and every subsequent request returns 503
until the pod is restarted.

The root issue is that not all exceptions indicate a broken model instance. A
`ValueError` caused by a malformed input, an `IndexError` from an edge-case in the
alignment analyser, or a `RuntimeError` from a degenerate tensor shape are
request-level failures — the slot itself is fine. Only hardware-level failures
(CUDA OOM, device assertion errors) are truly unrecoverable and warrant retiring the
slot.

---

## Solution: Two-tier error classification

### Tier 1 — Fatal errors (retire the slot immediately)

These indicate the GPU or the model process is in an unrecoverable state. No further
requests should be sent to this instance.

| Error type | Example message |
|---|---|
| `torch.cuda.OutOfMemoryError` | `CUDA out of memory. Tried to allocate...` |
| `torch.cuda.CudaError` | (any CUDA runtime error) |
| `RuntimeError` containing `"CUDA"` | device-side assertions, illegal memory access |
| `RuntimeError` containing `"NCCL"` | multi-GPU comms failure |

### Tier 2 — Transient errors (fail the request, keep the slot)

These are caused by input data or upstream library edge cases. The slot remains
healthy and is returned to the pool.

| Error type | Example |
|---|---|
| `IndexError` | alignment matrix empty for short text (the bug we just patched) |
| `ValueError` | bad tensor shape from unexpected input |
| `RuntimeError` (non-CUDA) | PyTorch numerical edge cases |
| `AssertionError` | internal model assertion on degenerate input |
| Everything else | unknown — default to keeping the slot |

Defaulting unknowns to **transient** is intentionally conservative. A slot that
survives a few requests after an unknown error proves it is healthy. If it is
actually broken, the consecutive-failure counter (see Phase 2 below) will retire it
after repeated failures.

---

## Implementation

### Phase 1 — Error classification (minimal change, high impact)

#### 1. Add `_is_fatal_generation_error` to `app/core/tts_model.py`

Place this near the top of the module, after the imports:

```python
def _is_fatal_generation_error(exc: Exception) -> bool:
    """Return True only for errors that leave the model instance in an unrecoverable state."""
    try:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
        # torch.cuda.CudaError exists in some torch versions
        cuda_error = getattr(torch.cuda, "CudaError", None)
        if cuda_error and isinstance(exc, cuda_error):
            return True
    except Exception:
        pass

    if isinstance(exc, RuntimeError):
        msg = str(exc).upper()
        return "CUDA" in msg or "NCCL" in msg

    return False
```

#### 2. Update `_generate_chunk_audio` in `app/api/endpoints/speech.py:354`

Current:
```python
    except Exception as exc:
        lease.mark_broken(str(exc))
        raise
```

Replacement:
```python
    except Exception as exc:
        if _is_fatal_generation_error(exc):
            lease.mark_broken(str(exc))
        raise
```

`_is_fatal_generation_error` needs to be imported from `app.core.tts_model`.

That is the entire Phase 1 change. It is deliberately small. Transient errors now
fail the request with 500 but leave the slot in the pool, so the next request can
use it normally.

---

### Phase 2 — Consecutive failure counter (extra resilience)

Phase 1 handles the clear-cut cases. Phase 2 catches the case where an unknown
error is actually unrecoverable but not classified as fatal — the slot keeps
failing every request silently.

#### 1. Extend `ModelSlot` in `app/core/tts_model.py:72`

```python
@dataclass
class ModelSlot:
    instance_id: int
    model: Any
    device: str
    healthy: bool = True
    last_error: Optional[str] = None
    consecutive_failures: int = 0          # NEW
```

Add a config constant alongside the other pool config:

```python
# Retire a slot after this many consecutive non-fatal failures in a row.
# A successful request resets the counter to 0.
MAX_CONSECUTIVE_SLOT_FAILURES: int = 5
```

#### 2. Add `mark_soft_failure` to `ModelLease` in `app/core/tts_model.py:80`

```python
@dataclass
class ModelLease:
    instance_id: int
    model: Any
    device: str
    broken: bool = False
    soft_failure: bool = False             # NEW
    failure_reason: Optional[str] = None
    released: bool = False

    def mark_broken(self, reason: str):
        self.broken = True
        self.failure_reason = reason

    def mark_soft_failure(self, reason: str):   # NEW
        self.soft_failure = True
        self.failure_reason = reason
```

#### 3. Update `_generate_chunk_audio` in `app/api/endpoints/speech.py:354`

```python
    except Exception as exc:
        if _is_fatal_generation_error(exc):
            lease.mark_broken(str(exc))
        else:
            lease.mark_soft_failure(str(exc))
        raise
```

#### 4. Update `release_model_lease` in `app/core/tts_model.py:451`

Current logic after `if lease.broken:` block:
```python
    if slot.healthy and _available_model_ids is not None:
        _available_model_ids.put_nowait(lease.instance_id)
    observe_pool_status(get_pool_status())
```

Replacement:
```python
    if lease.soft_failure:
        slot.consecutive_failures += 1
        slot.last_error = lease.failure_reason
        if slot.consecutive_failures >= MAX_CONSECUTIVE_SLOT_FAILURES:
            slot.healthy = False
            _update_runtime_after_slot_failure(
                lease.instance_id,
                f"retired after {slot.consecutive_failures} consecutive failures: {lease.failure_reason}",
            )
            return
    else:
        slot.consecutive_failures = 0   # successful request resets the counter

    if slot.healthy and _available_model_ids is not None:
        _available_model_ids.put_nowait(lease.instance_id)
    observe_pool_status(get_pool_status())
```

---

## Summary of file changes

| File | Change |
|---|---|
| `app/core/tts_model.py` | Add `_is_fatal_generation_error()` |
| `app/core/tts_model.py` | Add `consecutive_failures` field to `ModelSlot` |
| `app/core/tts_model.py` | Add `mark_soft_failure()` and `soft_failure` field to `ModelLease` |
| `app/core/tts_model.py` | Update `release_model_lease` to handle soft failures and reset counter |
| `app/api/endpoints/speech.py` | Replace unconditional `mark_broken` with classified call |

Phase 1 alone eliminates the cascading-503 failure for all transient errors.
Phase 2 adds a safety net for unknown errors that turn out to be persistent.
Both phases are backwards-compatible with the existing pool shutdown and metrics logic.
