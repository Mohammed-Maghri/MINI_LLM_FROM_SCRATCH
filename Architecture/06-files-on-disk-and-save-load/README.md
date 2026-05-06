# Files on disk and save/load

This document is **only** about persistence: what each file contains and which functions read or write them.

---

## 1. `instructions.txt` (merge rules)

**Writer (training):** `main.py` inside `run_bpe_training()`:

```python
with open(instr_path, "w", encoding="utf-8") as f:
    for instruction in tokens_instruction:
        f.write(instruction + "\n")
```

**Meaning of each line:** One merge from BPE training. The string is exactly what was stored as `token_instruction` in `recur_func`: two symbols separated by a **single ASCII space** (for example `h e` meaning merge `"h"` and `"e"` into `"he"`).

**Reader (inference):** `read_text_file` strips each line; `encoder_` uses `instruction.split()` to recover left and right.

**Also written by:** `tokenizer/tokenizer_bundle.py` → `save_tokenizer(...)` writes the same lines again (UTF-8). After `run_bpe_training`, the file is therefore written twice in sequence with equivalent content paths (same final result if lists match).

---

## 2. `output_set.json` (primary vocabulary path in `.env`)

**Writer:** `write_into_json` from `tokenizer/opened_files.py`, called from `main.py` with path `FILE_GENERATION` (default in `.env`: `tokenized_data/output_set.json`).

**Content shape:**

```json
{
  "tokens_generated": {
    "<token string>": <integer id>,
    ...
  }
}
```

**Reader:** `open_reads_json_file` used throughout; `encoder.py` loads it when `vocab=None` and `FILE_GENERATION` is set.

---

## 3. `vocab.json` (mirror vocabulary)

**Purpose:** Human-facing duplicate of the same JSON object as `output_set.json` so you can treat “vocabulary” as `vocab.json` without renaming `FILE_GENERATION`.

**Writer:** `save_tokenizer` in `tokenizer/tokenizer_bundle.py`, path from `.env` key `VOCAB_JSON` (default `tokenized_data/vocab.json`).

**Reader:** There is no separate loader required; any JSON reader works. The project standardizes on the same inner `tokens_generated` dict.

---

## 4. `.env` (paths the code reads)

Relevant keys used by `main.py` and `training/data_prep.py` / `encoder.py`:

| Key | Role |
|-----|------|
| `DATA_SET_FILE` | JSON input for BPE training |
| `BPE_MAX_MERGES` | Upper bound on `recur_func` iterations |
| `FILE_GENERATION` | Where `output_set.json` lives |
| `INSTRUCTIONS_FILE` | Where `instructions.txt` lives |
| `VOCAB_JSON` | Duplicate vocab JSON path |
| `DATA_SET_FOLDER` | Directory scanned for Phase 3 JSON |
| `PHASE3_SMOKE_JSON` | Preferred filename for quick Phase 3 smoke tests |

`python-dotenv` loads these at import time in modules that call `load_dotenv()`.

---

## 5. API-only helpers (no new on-disk format)

| Function | File | Role |
|----------|------|------|
| `save_tokenizer` | `tokenizer/tokenizer_bundle.py` | Writes vocab JSON + instructions UTF-8 |
| `load_raw_vocab` | same | Reads a vocab JSON file to dict |
| `load_instructions` | same | Reads instructions to `list[str]` |

These are thin wrappers around `json.load` / line iteration with UTF-8 encoding.

---

## Consistency rule

For correct encode/decode, the **merge list** and **vocabulary JSON** must come from the **same training run**. Mixing an old `instructions.txt` with a new `output_set.json` will produce meaningless ids or `KeyError` / UNK behavior.
