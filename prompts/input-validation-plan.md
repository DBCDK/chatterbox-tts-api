# Input Validation Plan: Minimum Text Length

## Problem

The chatterbox model is not designed for degenerate inputs. Very short text produces
an alignment matrix with too few columns for the repetition-detection heuristics in
`AlignmentStreamAnalyzer.step`. The `IndexError` we patched
(`prompts/exception-handling-plan.md`) is one symptom; other edge-case model
behaviour for near-empty inputs is likely.

The service currently validates only the *maximum* text length
(`app/api/endpoints/speech.py:261`). Both failing requests in the incident logs had
`input_chars: 1` — a single character that would never produce meaningful speech and
should be rejected at the API boundary before the model is ever touched.

---

## Solution: Add `MIN_TEXT_LENGTH` validation

Extend `_validate_text_length` to also enforce a minimum, controlled by a new
`MIN_TEXT_LENGTH` config value. Requests below the minimum receive a `400 Bad
Request` immediately, identical in shape to the existing too-long error.

---

## Implementation

### 1. Add `MIN_TEXT_LENGTH` to `app/config.py`

Place alongside the existing length constants (around line 28):

```python
MIN_TEXT_LENGTH: int = int(os.getenv("MIN_TEXT_LENGTH", "10"))
```

A default of `10` characters is conservative enough to rule out all degenerate
single-token inputs while still allowing very short phrases like `"OK"` at 2 chars
if the operator lowers the threshold via env var. Adjust the default to taste.

Add a validation rule in `Config.validate()` alongside the existing length checks
(around line 164):

```python
if cls.MIN_TEXT_LENGTH < 1:
    raise ValueError(
        f"MIN_TEXT_LENGTH must be at least 1, got {cls.MIN_TEXT_LENGTH}"
    )
if cls.MIN_TEXT_LENGTH >= cls.MAX_TOTAL_LENGTH:
    raise ValueError(
        f"MIN_TEXT_LENGTH ({cls.MIN_TEXT_LENGTH}) must be less than "
        f"MAX_TOTAL_LENGTH ({cls.MAX_TOTAL_LENGTH})"
    )
```

### 2. Extend `_validate_text_length` in `app/api/endpoints/speech.py:261`

Current:
```python
def _validate_text_length(text: str, mode: Optional[str] = None):
    if len(text) > Config.MAX_TOTAL_LENGTH:
        if mode is not None:
            observe_request_failure("input_too_long", "validation", mode)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": f"Input text too long. Maximum {Config.MAX_TOTAL_LENGTH} characters allowed.",
                    "type": "invalid_request_error",
                }
            },
        )
```

Replacement:
```python
def _validate_text_length(text: str, mode: Optional[str] = None):
    if len(text) < Config.MIN_TEXT_LENGTH:
        if mode is not None:
            observe_request_failure("input_too_short", "validation", mode)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": (
                        f"Input text too short. Minimum {Config.MIN_TEXT_LENGTH} characters required."
                    ),
                    "type": "invalid_request_error",
                }
            },
        )

    if len(text) > Config.MAX_TOTAL_LENGTH:
        if mode is not None:
            observe_request_failure("input_too_long", "validation", mode)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": f"Input text too long. Maximum {Config.MAX_TOTAL_LENGTH} characters allowed.",
                    "type": "invalid_request_error",
                }
            },
        )
```

The minimum check is placed first so both error paths are obvious when reading
the function.

### 3. Expose `min_text_length` in the health endpoint (`app/api/endpoints/health.py:60`)

Add alongside `max_chunk_length` and `max_total_length` so API consumers can
discover the limits at runtime:

```python
"min_text_length": Config.MIN_TEXT_LENGTH,
"max_chunk_length": Config.MAX_CHUNK_LENGTH,
"max_total_length": Config.MAX_TOTAL_LENGTH,
```

---

## Coverage

`_validate_text_length` is called at three sites, so all entry points are covered by
this single change:

| Call site | Route |
|---|---|
| `speech.py:373` | `_generate_full_audio` (internal path) |
| `speech.py:414` | `generate_speech_internal` (audio endpoint) |
| `speech.py:646` | `text_to_speech` (OpenAI-compatible endpoint) |

---

## Summary of file changes

| File | Change |
|---|---|
| `app/config.py` | Add `MIN_TEXT_LENGTH` env var and validation rule |
| `app/api/endpoints/speech.py` | Add minimum check to `_validate_text_length` |
| `app/api/endpoints/health.py` | Expose `min_text_length` in health config response |
