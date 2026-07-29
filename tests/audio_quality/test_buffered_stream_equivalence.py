"""Phase 1 Work Item 3 gate (see prompts/streaming-phase-1-plan.md).

Exit A (the buffered, non-streaming response) is supposed to switch from calling
``ChatterboxInference.generate()`` directly to draining the same streaming source used
by Exit B/C (``generate_stream_async()``) and concatenating the chunks -- that is what
collapses the two paths into one pipeline and removes the double buffer copy.

That rewrite is only safe if concatenating the streamed chunks reproduces generate()'s
output. The library's own docstring claims this ("Concatenating all yielded tensors
produces the same result as generate()."), but per the plan a docstring is not a
guarantee: if this test fails, Exit A must keep calling generate() directly and the two
paths stay split -- better to find that out here than in production audio.

Opt-in: set RUN_AUDIO_QUALITY_TESTS=1. Loads real model weights and runs real inference --
not part of the default fast unit-test run. Skip is a module-level skipif (collection
time), matching test_null_harness.py, for the same reason: this module's fixtures are
module-scoped and would otherwise load the model before a function-scoped autouse skip
fixture got a chance to run.
"""

import asyncio
import json
import os
from pathlib import Path

import pytest
import torch

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
# Multi-sentence on purpose: a single-sentence input can't exercise the inter-sentence
# silence insertion both code paths perform, which is exactly where they could diverge.
MULTI_SENTENCE_ENTRY = next(e for e in CORPUS if e["id"] == "lokalplan-01")

# voices/mic-voice.wav is checked into git (unlike vocoder_isolation.py's coral_sample.wav,
# which is a machine-local file on the Phase 0 GPU box, not tracked) -- this keeps the test
# runnable from a fresh checkout without an out-of-band asset.
DEFAULT_VOICE_SAMPLE = os.path.join(os.path.dirname(__file__), "..", "..", "voices", "mic-voice.wav")
SEED = 0


@pytest.fixture(scope="module")
def loaded_inference() -> ChatterboxInference:
    model = ChatterboxInference.from_pretrained(
        model_type="multilingual",
        language="da",
        sentence_split=True,
        inter_sentence_silence_ms=100,
    )
    model.prepare_conditionals(DEFAULT_VOICE_SAMPLE)
    return model


def test_concatenated_stream_matches_generate(loaded_inference):
    text = MULTI_SENTENCE_ENTRY["text"]

    torch.manual_seed(SEED)
    whole = loaded_inference.generate(text, language_id="da")

    async def _drain_stream():
        chunks = [
            chunk
            async for chunk in loaded_inference.generate_stream_async(text, language_id="da")
        ]
        return torch.cat(chunks, dim=-1)

    torch.manual_seed(SEED)
    streamed = asyncio.run(_drain_stream())

    assert whole.shape == streamed.shape, (
        f"shape mismatch: generate()={tuple(whole.shape)} "
        f"vs concatenated stream={tuple(streamed.shape)}"
    )
    max_abs_diff = (whole - streamed).abs().max().item()
    assert torch.allclose(whole, streamed, atol=1e-4, rtol=1e-4), (
        f"generate() and concatenated generate_stream_async() diverged "
        f"(max abs diff {max_abs_diff:.6f}) -- Exit A must keep calling generate() "
        f"directly rather than draining the streaming source; the two paths cannot be "
        f"collapsed as currently implemented."
    )
