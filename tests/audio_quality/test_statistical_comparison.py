"""Tier 2 null test: end-to-end statistical comparison (Phase 0 Deliverable 4).

Opt-in: set RUN_AUDIO_QUALITY_TESTS=1. Loads real model weights and runs real inference --
not part of the default fast unit-test run. See test_null_harness.py for why this is a
module-level skipif rather than an autouse fixture.
"""

import json
import os
from pathlib import Path

import pytest
import torch

from tests.audio_quality import statistical_comparison as sc
from tests.audio_quality import vocoder_isolation as vi
from chatterbox.inference import ChatterboxInference

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
TEST_ENTRY = next(e for e in CORPUS if e["id"] == "lokalplan-01")
SEED = 42
SR = 24000  # chatterbox multilingual model's synthesis rate


@pytest.fixture(scope="module")
def inference_model():
    model = ChatterboxInference.from_pretrained(
        model_type="multilingual",
        language="da",
        device=None,
        normalize_text=True,
        sentence_split=True,
        inter_sentence_silence_ms=100,
    )
    model.prepare_conditionals(vi.DEFAULT_VOICE_SAMPLE)
    return model


def _generate(model, seed):
    torch.manual_seed(seed)
    return model.generate(TEST_ENTRY["text"], language_id="da")


@pytest.fixture(scope="module")
def reference_wav(inference_model):
    return _generate(inference_model, SEED)


def test_repeat_generation_matches_reference(inference_model, reference_wav):
    """Same seed, same text -> fingerprints should match closely. This is the
    self-comparison baseline: if this doesn't pass, the comparison function itself (not
    the model) has a bug."""
    repro_wav = _generate(inference_model, SEED)

    ref_fp = sc.compute_fingerprint(reference_wav, SR)
    repro_fp = sc.compute_fingerprint(repro_wav, SR)
    diff = sc.compare_fingerprints(ref_fp, repro_fp)

    assert ref_fp.sentences, "expected the multi-sentence lokalplan text to produce sentence segments"
    assert diff.duration_diff_s < 0.02
    assert diff.sentence_count_diff == 0
    assert diff.silence_gap_count_diff == 0
    assert all(g < 20 for g in diff.silence_gap_ms_diffs), diff.silence_gap_ms_diffs
    assert all(c < 0.01 for c in diff.mel_cosine_distances), diff.mel_cosine_distances


def test_harness_detects_missing_inter_sentence_silence(reference_wav):
    """Proof the gate can fail: silently dropping the 100ms inter-sentence silence (a
    real, plausible regression -- e.g. a Phase 3 chunk-boundary bug that eats the gap)
    must be caught by the silence-gap-structure check."""
    ref_fp = sc.compute_fingerprint(reference_wav, SR)
    gaps = sc.find_silence_gaps(reference_wav.squeeze(), SR)
    assert gaps, "expected the lokalplan text to produce at least one inter-sentence gap"

    corrupted = torch.cat(
        sc.split_on_silence(reference_wav.squeeze(), SR, gaps)
    ).unsqueeze(0)
    corrupted_fp = sc.compute_fingerprint(corrupted, SR)
    diff = sc.compare_fingerprints(ref_fp, corrupted_fp)

    assert diff.silence_gap_count_diff > 0 or diff.duration_diff_s > 0.02, (
        "expected removing inter-sentence silence to be caught by the statistical "
        f"comparison, but diff looked clean: {diff}"
    )
