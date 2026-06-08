"""Runtime patch for installed chatterbox package behavior."""

from __future__ import annotations

_PATCHED = False


def apply_chatterbox_patches() -> None:
    """Patch AlignmentStreamAnalyzer to handle short text inputs gracefully."""
    global _PATCHED
    if _PATCHED:
        return

    from chatterbox.models.t3.inference.alignment_stream_analyzer import (
        AlignmentStreamAnalyzer,
    )

    _original_step = AlignmentStreamAnalyzer.step

    def _patched_step(self, logits, next_token=None):
        try:
            return _original_step(self, logits, next_token)
        except IndexError as exc:
            if "Expected reduction dim 1 to have non-zero size" in str(exc):
                # Text too short for repetition check — skip EOS forcing and continue.
                return logits
            raise

    AlignmentStreamAnalyzer.step = _patched_step
    _PATCHED = True
