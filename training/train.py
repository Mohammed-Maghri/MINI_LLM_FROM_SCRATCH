"""
Minimal training tensor demo: LM batches from Phase 3 data preparation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from tokenizer.opened_files import open_reads_json_file
from tokenizer.tokenizer_bundle import load_instructions
from training.data_prep import prepare_training_bundle

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    vocab_path = ROOT / os.getenv("FILE_GENERATION", "tokenized_data/output_set.json")
    instr_path = ROOT / os.getenv("INSTRUCTIONS_FILE", "tokenized_data/instructions.txt")
    data_dir = ROOT / os.getenv("DATA_SET_FOLDER", "data_set")

    base_vocab = open_reads_json_file(str(vocab_path))
    instructions = load_instructions(str(instr_path))
    json_files = sorted(data_dir.glob("*.json"))
    if not json_files:
        raise SystemExit("No JSON files in data_set folder.")

    x, y, vocab_ext, specials = prepare_training_bundle(
        json_paths=json_files[:1],
        text_glob=None,
        instructions=instructions,
        base_vocab=base_vocab,
        seq_len=8,
        batch_size=2,
    )

    vocab_size = len(vocab_ext.get("tokens_generated", {}))
    emb = np.random.randn(vocab_size, 64) / np.sqrt(64)

    print("Batch X shape:", x.shape)
    print("Batch Y shape:", y.shape)
    print("Embedding table rows:", emb.shape[0], "specials:", specials)


if __name__ == "__main__":
    main()
