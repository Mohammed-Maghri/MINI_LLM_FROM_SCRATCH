# Phase 3 — Corpus collection and the flat token-id stream

Source file: `training/data_prep.py`

This file does **not** train BPE. It assumes merges and base vocab already exist.

---

## Part A — `load_json_word_list(path) -> list[str]`

**Steps:**

1. Open the path as UTF-8 JSON.

2. If the root JSON value is a **list**, return `[str(x) for x in data]`.

3. Else if the object has key `"data_set"` whose value is a list, return each element as `str(...)`.

4. Else if the object has key `"words"` whose value is a list, return each element as `str(...)`.

5. Else raise `ValueError` with “Unrecognized JSON schema”.

**Consequence:** Each JSON file contributes an ordered flat list of strings. Each string is one **logical segment** you will tokenize independently (often one word, sometimes a longer line depending on your JSON).

---

## Part B — `collect_corpus(json_paths=None, text_glob=None) -> list[str]`

**If `json_paths` is provided (non-empty list):**

For each path, extend the output list with `load_json_word_list(p)`.

**If `text_glob` is provided:**

1. Parse it as `Path(text_glob)`; split into `parent` directory and `pattern` filename glob.

2. `sorted(parent.glob(pattern))` enumerates matching files.

3. For each file, read full text UTF-8 with `errors="replace"`.

4. Split into lines; strip each line; **skip empty lines**; append non-empty lines as separate corpus strings.

**If both are provided:** JSON-derived strings come **first** in the combined list, then all text-file lines (each file processed in sorted path order).

**Output:** A single Python `list[str]` called “corpus” at the Python level (still plain text, not ids yet).

---

## Part C — `tokenize_corpus(...) -> list[int]`

**Parameters:**

- `texts: list[str]` — usually the return value of `collect_corpus`.
- `instructions: list[str]` — merge lines (same format as `encoder_` expects).
- `vocab: dict` — must include `"tokens_generated"`; in Phase 3 this is typically the **extended** vocab (base merges + special tokens); see `09-special-tokens.md`.
- `specials: SpecialTokenIds` — dataclass with `.bos`, `.eos`, `.unk` (and `.pad`).
- `bos_every: bool` — default `False`.
- `eos_between: bool` — default `True`.

**Loop over `enumerate(texts)`:**

1. If `bos_every` is true, append `specials.bos` to `all_ids`.

2. Extend `all_ids` with:

   ```python
   encode_to_ids(text, instructions=instructions, vocab=vocab, unk_id=specials.unk)
   ```

   So every character-level / merge-level piece that is **not** in `tokens_generated` becomes the integer `specials.unk` instead of raising.

3. If `eos_between` is true, append `specials.eos` after each text segment (including the last one).

**Output:** One long Python list of integers representing the entire corpus in order, with optional BOS markers and EOS markers between segments.

This list is the input to batching (`make_lm_batches`, next document).
