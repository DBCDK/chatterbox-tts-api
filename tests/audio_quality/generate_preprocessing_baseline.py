"""Snapshot what the current coral chatterbox stack's text preprocessing produces.

This is the Phase 2 acceptance gate (see prompts/streaming-phase-0-plan.md, Deliverable 2) and
must be regenerated only while the pre-Phase-2 `chatterbox` dependency is still installed --
once it is swapped for pinned upstream ResembleAI/chatterbox, this script's imports go away
along with the behaviour it is recording.

Usage: uv run python tests/audio_quality/generate_preprocessing_baseline.py
"""

import json
from pathlib import Path

from chatterbox.utils.normalizer import normalize_text
from chatterbox.utils.splitter import split_sentences

from app.config import Config

LANGUAGE = "da"
CORPUS_PATH = Path(__file__).parent / "corpus.json"
OUTPUT_PATH = Path(__file__).parent / "preprocessing_baseline.json"


def main() -> None:
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))

    snapshot = []
    for entry in corpus:
        text = entry["text"]
        normalized = normalize_text(text, language=LANGUAGE) if Config.NORMALIZE_TEXT else text
        sentences = split_sentences(normalized, language=LANGUAGE)
        snapshot.append(
            {
                "id": entry["id"],
                "input": text,
                "normalized": normalized,
                "sentences": sentences,
            }
        )

    OUTPUT_PATH.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(snapshot)} entries to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
