"""Runtime patches for installed chatterbox package behavior."""

from __future__ import annotations

import json
from pathlib import Path


_PATCHED = False


def _candidate_cangjie_paths(model_dir: Path) -> list[Path]:
    resolved_dir = model_dir.resolve()
    candidate_roots = []
    for root in (model_dir, resolved_dir):
        if root not in candidate_roots:
            candidate_roots.append(root)

    candidates: list[Path] = []
    for root in candidate_roots:
        direct_file = root / "Cangjie5_TC.json"
        if direct_file not in candidates:
            candidates.append(direct_file)

        repo_cache_root = root / "models--ResembleAI--chatterbox"
        refs_main = repo_cache_root / "refs" / "main"
        if refs_main.exists():
            revision = refs_main.read_text(encoding="utf-8").strip()
            if revision:
                snapshot_file = (
                    repo_cache_root / "snapshots" / revision / "Cangjie5_TC.json"
                )
                if snapshot_file not in candidates:
                    candidates.append(snapshot_file)

    return candidates


def apply_chatterbox_patches() -> None:
    """Apply local-first fixes to the installed chatterbox package."""
    global _PATCHED
    if _PATCHED:
        return

    from chatterbox.models.tokenizers import tokenizer as tokenizer_module

    converter_cls = tokenizer_module.ChineseCangjieConverter
    original_loader = converter_cls._load_cangjie_mapping

    def _load_cangjie_mapping(self, model_dir=None):
        self.word2cj = {}
        self.cj2word = {}

        if model_dir is not None:
            for candidate in _candidate_cangjie_paths(Path(model_dir)):
                if not candidate.exists():
                    continue

                with candidate.open("r", encoding="utf-8") as fp:
                    data = json.load(fp)

                for entry in data:
                    word, code = entry.split("\t")[:2]
                    self.word2cj[word] = code
                    if code not in self.cj2word:
                        self.cj2word[code] = [word]
                    else:
                        self.cj2word[code].append(word)
                return

        return original_loader(self, model_dir)

    converter_cls._load_cangjie_mapping = _load_cangjie_mapping
    _PATCHED = True
