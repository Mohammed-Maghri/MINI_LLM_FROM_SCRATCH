from __future__ import annotations

import os
from typing import Iterable

from dotenv import load_dotenv

from tokenizer.opened_files import open_reads_json_file, read_text_file

load_dotenv()


def _instructions_path(explicit: list[str] | None) -> list[str]:
    if explicit is not None:
        return explicit
    path = os.getenv("INSTRUCTIONS_FILE")
    if not path:
        return []
    return read_text_file(path)


def _vocab_dict(explicit: dict | None) -> dict:
    if explicit is not None:
        return explicit
    path = os.getenv("FILE_GENERATION")
    if not path:
        return {}
    return open_reads_json_file(path)


def encoder_(word: str, instructions: list[str] | None = None) -> list[str]:
    """
    Apply learned BPE merge rules to text. Returns subword token strings
    (not numeric ids).
    """
    instructions = _instructions_path(instructions)
    splited_word: list[str] = [word[i] for i in range(len(word))]

    for instruction in instructions:
        encoded_word = instruction.split()
        if len(encoded_word) != 2:
            continue
        index = 0
        while index < len(splited_word) - 1:
            if splited_word[index] == encoded_word[0] and splited_word[index + 1] == encoded_word[1]:
                splited_word[index] = encoded_word[0] + encoded_word[1]
                del splited_word[index + 1]
            else:
                index += 1
    return splited_word


def decode_tokens(
    encoded_tokens: list[str],
    vocab: dict | None = None,
    unk_id: int | None = None,
) -> list[int]:
    """Map subword token strings to ids. Unknown tokens resolve to unk_id when set."""
    vocab = _vocab_dict(vocab)
    table = vocab.get("tokens_generated", {})
    out: list[int] = []
    for token in encoded_tokens:
        tid = table.get(token)
        if tid is None:
            if unk_id is None:
                raise KeyError(f"Token not in vocabulary: {token!r}")
            out.append(unk_id)
        else:
            out.append(tid)
    return out


def _id_to_token(vocab: dict) -> dict[int, str]:
    table = vocab.get("tokens_generated", {})
    rev: dict[int, str] = {}
    for tok, i in table.items():
        rev[i] = tok
    return rev


def decoder(token_ids: Iterable[int], vocab: dict | None = None) -> str:
    """Concatenate decoded subwords (BPE-style, no spaces between pieces)."""
    vocab = _vocab_dict(vocab)
    rev = _id_to_token(vocab)
    parts: list[str] = []
    for i in token_ids:
        parts.append(rev.get(int(i), ""))
    return "".join(parts)


def encode_to_ids(
    text: str,
    *,
    instructions: list[str] | None = None,
    vocab: dict | None = None,
    unk_id: int | None = None,
) -> list[int]:
    """Full pipeline: text -> BPE tokens -> token ids."""
    toks = encoder_(text, instructions=instructions)
    return decode_tokens(toks, vocab=vocab, unk_id=unk_id)
