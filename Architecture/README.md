# Architecture notes

This folder explains **exactly** what the codebase does. Each topic lives in **its own subfolder** with a `README.md`. Use the links below (the **bold text** is what you click—each goes to that topic’s explanation).

## Browse by topic

- [**Overview and three pipelines**](01-overview-and-data-flow/README.md) — training BPE vs encoding text vs preparing LM data; dependency chain.
- [**Phase 1 — BPE training (learning merges)**](02-phase1-bpe-training/README.md) — JSON → character lists, pair counts, one merge at a time, `recur_func` until cap or stop.
- [**Building the vocabulary table (token → id)**](03-building-the-vocabulary-table/README.md) — Latin-1 byte slots and merge-order ids starting at 256.
- [**Phase 2 — Encoding text to subwords**](04-phase2-encoding-text-to-subwords/README.md) — `encoder_`, merge line format, inner/outer loops.
- [**Phase 2 — Subwords to ids and back**](05-phase2-subwords-to-ids-and-back/README.md) — `decode_tokens`, `decoder`, `encode_to_ids`, UNK and round-trip caveats.
- [**Files on disk and save/load**](06-files-on-disk-and-save-load/README.md) — what each file stores, UTF-8 vs env paths, consistency rules.
- [**Phase 3 — Corpus and flat token stream**](07-phase3-corpus-and-token-stream/README.md) — JSON and text glob ingestion, `tokenize_corpus`, BOS/EOS between strings.
- [**Phase 3 — Language-model batches**](08-phase3-language-model-batches/README.md) — sliding windows into `(X, Y)`, reshape to `(batches, batch_size, seq_len)`.
- [**Special tokens (PAD / BOS / EOS / UNK)**](09-special-tokens/README.md) — string names, contiguous ids after max vocab id, where they are used.
- [**Entry points (`main`, training modules, tests)**](10-entry-points-main-and-modules/README.md) — `main.py train_bpe`, `python -m training.data_prep`, `training/train`, `tests/test_tokenizer`.

## Reading order (same links)

If you prefer a fixed order, read **1 → 10** using the list above: start with [Overview and three pipelines](01-overview-and-data-flow/README.md), then follow the bullets top to bottom.

Together, these match the implementation under `tokenizer/` and `training/` as of this documentation set.
