# Entry points: `main`, training modules, tests

This document lists **how to run** each part and **what file** owns the behavior.

---

## `main.py`

### Default mode (`python main.py` with no arguments)

1. `load_dotenv()` reads `.env`.

2. Reads `ENCODE_SAMPLE` (default `"journey"`).

3. Calls `encode_to_ids(sample)` — uses env paths for instructions and vocab.

4. Calls `decoder(ids)` — same default vocab path.

5. Prints the ids and decoded string, then prints hints for other commands.

**Purpose:** Quick sanity check that tokenizer assets on disk match the code paths.

### `python main.py train_bpe`

Calls `run_bpe_training()`:

1. `open_reads_json_file(os.getenv("DATA_SET_FILE", ...))`.

2. `start_word_set_generation(opened_data)`.

3. `recur_func(tokens_saved, tokens_instruction, word_sets, 0, int(os.getenv("BPE_MAX_MERGES", "3000")))`.

4. `vocab = generate_tokens_list(tokens_saved)`.

5. `write_into_json(FILE_GENERATION, vocab)`.

6. Writes `INSTRUCTIONS_FILE` line by line from `tokens_instruction`.

7. `save_tokenizer(vocab, tokens_instruction, vocab_json_path=VOCAB_JSON, instructions_path=INSTRUCTIONS_FILE)`.

**Side effect:** Global lists `tokens_saved` and `tokens_instruction` in `main.py` are appended across runs unless you restart the interpreter or clear them. For one-shot CLI use this is fine; for repeated training in one process, reset those lists.

---

## `python -m training.data_prep`

When executed as `__main__` in `training/data_prep.py`:

1. Adds project root to `sys.path` if missing.

2. `load_dotenv()`.

3. Resolves paths: `FILE_GENERATION`, `INSTRUCTIONS_FILE`, `DATA_SET_FOLDER`, `PHASE3_SMOKE_JSON`.

4. Picks JSON files: glob `PHASE3_SMOKE_JSON` inside `DATA_SET_FOLDER`; if none, falls back to first `*.json` in that folder.

5. Loads vocab and instructions.

6. Calls `prepare_training_bundle(...)` with fixed demo `seq_len=8`, `batch_size=4`.

7. Prints shapes and special token ids.

**Purpose:** Smoke test Phase 3 without importing a heavy training loop.

---

## `python -m training.train`

File: `training/train.py`

1. Inserts project root on `sys.path`.

2. Loads vocab + instructions like data prep.

3. Uses **only** `sorted(data_dir.glob("*.json"))[:1]` — the first JSON file in lexicographic order under `data_set/` (often `common_words_1000.json` before `data.json`).

4. `prepare_training_bundle` with `seq_len=8`, `batch_size=2`.

5. Builds `np.random.randn(vocab_size, 64) / np.sqrt(64)` as a toy embedding matrix row count matching extended vocab.

**Purpose:** Demonstrate that batch tensors exist and embedding width can be chosen independently of vocab size.

---

## `tests/test_tokenizer.py`

Standard library `unittest`.

- Changes cwd to project root in `setUpClass` and loads `.env`.

- Tests: string output type of `encoder_`, strict round-trip for `"journey"`, consistency of `encode_to_ids` vs manual pipeline, UNK path with extended vocab.

**Run:**

```bash
python -m unittest tests.test_tokenizer -v
```

(from repository root, with dependencies installed)

---

## Optional: PDF ingestion

`tokenizer/data_set_loader.py` contains helpers (`extract_text_from_files`) that can append normalized words into `data_set/data.json` using PyMuPDF. That path is **separate** from the three numbered architecture pipelines unless you explicitly run it to grow your training JSON.
