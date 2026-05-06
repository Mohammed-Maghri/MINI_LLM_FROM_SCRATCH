"""Reserved vocabulary IDs for training and batching (Phase 3)."""

from __future__ import annotations

import copy
from dataclasses import dataclass


PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"


@dataclass(frozen=True)
class SpecialTokenIds:
    pad: int
    bos: int
    eos: int
    unk: int


def _max_token_id(tokens_generated: dict) -> int:
    if not tokens_generated:
        return -1
    return max(tokens_generated.values())


def extend_vocab_with_specials(vocab: dict) -> tuple[dict, SpecialTokenIds]:
    """
    Return a new vocab dict (deep copy of tokens_generated) with four special
    entries appended after the current maximum id.
    """
    inner = copy.deepcopy(vocab.get("tokens_generated", {}))
    start = _max_token_id(inner) + 1
    inner[PAD_TOKEN] = start
    inner[BOS_TOKEN] = start + 1
    inner[EOS_TOKEN] = start + 2
    inner[UNK_TOKEN] = start + 3
    ids = SpecialTokenIds(pad=start, bos=start + 1, eos=start + 2, unk=start + 3)
    return {"tokens_generated": inner}, ids


def vocab_size(vocab: dict) -> int:
    return len(vocab.get("tokens_generated", {}))
