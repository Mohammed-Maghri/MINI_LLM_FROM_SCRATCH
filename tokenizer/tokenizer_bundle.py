"""Save and load tokenizer assets: merge rules + vocabulary JSON."""

from __future__ import annotations

import json
from pathlib import Path


def save_tokenizer(
    vocab: dict,
    instructions: list[str],
    *,
    vocab_json_path: str,
    instructions_path: str,
) -> None:
    """
    Persist vocabulary (token -> id) and one merge rule per line (two symbols,
    space-separated). Also mirrors the vocabulary to vocab_json_path for the
    Phase 2 contract (instructions.txt + vocab.json).
    """
    Path(vocab_json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(instructions_path).parent.mkdir(parents=True, exist_ok=True)

    with open(vocab_json_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)

    with open(instructions_path, "w", encoding="utf-8") as f:
        for line in instructions:
            f.write(line.rstrip("\n") + "\n")


def load_raw_vocab(vocab_json_path: str) -> dict:
    with open(vocab_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_instructions(instructions_path: str) -> list[str]:
    with open(instructions_path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f.readlines()]
