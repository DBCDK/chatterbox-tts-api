"""Tier 1 null test: vocoder isolation (Phase 0 Deliverable 4).

Opt-in: set RUN_AUDIO_QUALITY_TESTS=1. Loads real model weights and runs real inference --
not part of the default fast unit-test run.

The skip below is a module-level skipif (collection-time), not an autouse fixture --
this module's fixtures are module-scoped and would otherwise be set up (loading the
model, generating tokens) *before* a function-scoped autouse skip fixture runs, since
pytest instantiates higher-scoped fixtures first regardless of skip decisions made by a
narrower-scoped one.
"""

import json
import os
from pathlib import Path

import pytest

from tests.audio_quality import vocoder_isolation as vi

RUN_AUDIO_QUALITY_TESTS = os.getenv("RUN_AUDIO_QUALITY_TESTS", "").lower() in ("1", "true")

pytestmark = [
    pytest.mark.audio_quality,
    pytest.mark.skipif(
        not RUN_AUDIO_QUALITY_TESTS,
        reason="Set RUN_AUDIO_QUALITY_TESTS=1 to run tests that load real model weights.",
    ),
]

CORPUS_PATH = Path(__file__).parent / "corpus.json"
CORPUS = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
TEST_ENTRY = next(e for e in CORPUS if e["id"] == "abbrev-01")

# Empirically derived (see vocoder_isolation.py docstring): overlap=10 keeps every
# boundary within ~1% of the reference peak; overlap=0 (naive, no context) blows past it.
BOUNDARY_EPSILON = 0.05
SEGMENTAL_SNR_FLOOR_DB = 15.0


@pytest.fixture(scope="module")
def loaded_model():
    return vi.load_model()


@pytest.fixture(scope="module")
def fixed_speech_tokens(loaded_model):
    return vi.generate_speech_tokens(loaded_model, TEST_ENTRY["text"])


@pytest.fixture(scope="module")
def whole_reference(loaded_model, fixed_speech_tokens):
    return vi.synthesize_whole(loaded_model, fixed_speech_tokens)


def test_chunked_vocoder_matches_whole_sequence(loaded_model, whole_reference):
    """The real gate: chunked-with-context-overlap must reproduce whole-sequence
    synthesis closely at every chunk boundary, not just on average."""
    whole_mel, whole_wav = whole_reference
    chunked_wav, boundaries = vi.synthesize_chunked(
        loaded_model, whole_mel, mel_chunk_frames=40, overlap_frames=10
    )

    assert boundaries, "expected at least one chunk boundary for this corpus entry"

    deviations = vi.boundary_max_deviation(whole_wav, chunked_wav, boundaries)
    for d in deviations:
        assert d["max_deviation"] < BOUNDARY_EPSILON, (
            f"boundary@{d['boundary']} deviated {d['max_deviation']:.4f} "
            f"(ref_peak={d['ref_peak']:.4f}, epsilon={BOUNDARY_EPSILON})"
        )

    snrs = vi.segmental_snr_db(whole_wav, chunked_wav)
    assert snrs, "expected at least one non-silent segment to compare"
    assert min(snrs) > SEGMENTAL_SNR_FLOOR_DB, f"segmental SNRs: {snrs}"


def test_harness_detects_corrupted_chunk_boundary(loaded_model, whole_reference):
    """Proof the gate can fail: dropping the overlap context (the naive, no-look-ahead
    chunking this harness exists to catch) must blow past the same threshold."""
    whole_mel, whole_wav = whole_reference
    corrupted_wav, boundaries = vi.synthesize_chunked(
        loaded_model, whole_mel, mel_chunk_frames=40, overlap_frames=0
    )

    deviations = vi.boundary_max_deviation(whole_wav, corrupted_wav, boundaries)
    assert any(d["max_deviation"] >= BOUNDARY_EPSILON for d in deviations), (
        "expected the corrupted (no-overlap) chunking to fail the boundary check "
        f"but all deviations were under {BOUNDARY_EPSILON}: {deviations}"
    )
