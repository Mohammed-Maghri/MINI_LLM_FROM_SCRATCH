"""
Phase 3: collect a text corpus, tokenize to ids, add special tokens, and build
next-token prediction batches.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tokenizer.encoder import encode_to_ids
from tokenizer.special_tokens import SpecialTokenIds, extend_vocab_with_specials


def load_json_word_list(path: str | Path) -> list[str]:
    """Load strings from data_set-style JSON (data_set, words, or word list root)."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [str(x) for x in data]
    if "data_set" in data and isinstance(data["data_set"], list):
        return [str(x) for x in data["data_set"]]
    if "words" in data and isinstance(data["words"], list):
        return [str(x) for x in data["words"]]
    raise ValueError(f"Unrecognized JSON schema in {path}")


def collect_corpus(
    json_paths: list[str | Path] | None = None,
    text_glob: str | Path | None = None,
) -> list[str]:
    """
    Aggregate training lines/strings from JSON corpora and optional plain-text files.

    - json_paths: list of .json files (each word/line becomes one string to tokenize).
    - text_glob: e.g. ``corpus/*.txt`` — each non-empty line is one string.
    """
    out: list[str] = []
    if json_paths:
        for p in json_paths:
            out.extend(load_json_word_list(p))
    if text_glob:
        base = Path(text_glob)
        parent, pattern = base.parent, base.name
        for fp in sorted(parent.glob(pattern)):
            if fp.is_file():
                raw = fp.read_text(encoding="utf-8", errors="replace")
                for line in raw.splitlines():
                    line = line.strip()
                    if line:
                        out.append(line)
    return out


def tokenize_corpus(
    texts: list[str],
    *,
    instructions: list[str],
    vocab: dict,
    specials: SpecialTokenIds,
    bos_every: bool = False,
    eos_between: bool = True,
) -> list[int]:
    """
    Encode each text segment to ids, inserting EOS between segments (and optional
    BOS at the start of each segment).
    """
    all_ids: list[int] = []
    unk = specials.unk
    for i, text in enumerate(texts):
        if bos_every:
            all_ids.append(specials.bos)
        all_ids.extend(
            encode_to_ids(text, instructions=instructions, vocab=vocab, unk_id=unk)
        )
        if eos_between:
            all_ids.append(specials.eos)
    return all_ids


def make_lm_batches(
    token_ids: list[int],
    seq_len: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, Y) for next-token prediction at every offset: both shaped
    (num_batches, batch_size, seq_len). Y[:, b, t] == X[:, b, t+1] within each row;
    last target may wrap to next row's first token — for strict autoregressive blocks,
    use non-overlapping chunks instead; this matches common sliding-window LM data.
    """
    if seq_len < 2:
        raise ValueError("seq_len must be at least 2")
    ids = np.asarray(token_ids, dtype=np.int64)
    if len(ids) <= seq_len:
        raise ValueError("Need more than seq_len tokens for batching")

    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    for i in range(0, len(ids) - seq_len, seq_len):
        chunk = ids[i : i + seq_len + 1]
        if len(chunk) < seq_len + 1:
            break
        x_rows.append(chunk[:-1])
        y_rows.append(chunk[1:])

    if not x_rows:
        raise ValueError("Corpus too short for one batch row")

    x_stack = np.stack(x_rows, axis=0)
    y_stack = np.stack(y_rows, axis=0)
    n = (x_stack.shape[0] // batch_size) * batch_size
    x_stack = x_stack[:n]
    y_stack = y_stack[:n]
    if n == 0:
        raise ValueError("Not enough rows for batch_size")

    new_shape = (n // batch_size, batch_size, seq_len)
    return x_stack.reshape(new_shape), y_stack.reshape(new_shape)


def prepare_training_bundle(
    *,
    json_paths: list[str | Path] | None = None,
    text_glob: str | Path | None = None,
    instructions: list[str],
    base_vocab: dict,
    seq_len: int,
    batch_size: int,
    bos_every: bool = False,
    eos_between: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict, SpecialTokenIds]:
    """
    End-to-end Phase 3: corpus -> extended vocab with specials -> token ids -> batches.
    """
    vocab_ext, specials = extend_vocab_with_specials(base_vocab)
    texts = collect_corpus(json_paths=json_paths, text_glob=text_glob)
    if not texts:
        raise ValueError("No corpus texts collected; check paths and glob.")
    ids = tokenize_corpus(
        texts,
        instructions=instructions,
        vocab=vocab_ext,
        specials=specials,
        bos_every=bos_every,
        eos_between=eos_between,
    )
    x, y = make_lm_batches(ids, seq_len=seq_len, batch_size=batch_size)
    return x, y, vocab_ext, specials


if __name__ == "__main__":
    import os
    import sys

    from dotenv import load_dotenv

    from tokenizer.opened_files import open_reads_json_file
    from tokenizer.tokenizer_bundle import load_instructions

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    load_dotenv()
    vocab_path = root / os.getenv("FILE_GENERATION", "tokenized_data/output_set.json")
    instr_path = root / os.getenv("INSTRUCTIONS_FILE", "tokenized_data/instructions.txt")
    data_dir = root / os.getenv("DATA_SET_FOLDER", "data_set")
    prefer = os.getenv("PHASE3_SMOKE_JSON", "common_words_1000.json")
    json_files = sorted(data_dir.glob(prefer))
    if not json_files:
        json_files = sorted(data_dir.glob("*.json"))[:1]

    base_vocab = open_reads_json_file(str(vocab_path))
    instructions = load_instructions(str(instr_path))

    x, y, v_ext, sp = prepare_training_bundle(
        json_paths=json_files,
        text_glob=None,
        instructions=instructions,
        base_vocab=base_vocab,
        seq_len=8,
        batch_size=4,
    )
    print("Phase 3 smoke test")
    print("X batch shape:", x.shape, "Y batch shape:", y.shape)
    print("Special ids:", sp)
    print("Extended vocab size:", len(v_ext.get("tokens_generated", {})))
