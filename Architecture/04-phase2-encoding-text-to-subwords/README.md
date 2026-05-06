# Phase 2 — Encoding: text to subword strings (`encoder_`)

Source file: `tokenizer/encoder.py`  
Function: `encoder_(word: str, instructions: list[str] | None = None) -> list[str]`

---

## What this function returns

A **list of Python strings**. Each element is one **subword token** after BPE-style merging.

It does **not** return integers. Integer assignment is `decode_tokens` (next document).

---

## Step 0 — Resolve merge rules

If `instructions is None`:

1. Read `os.getenv("INSTRUCTIONS_FILE")`.
2. Call `read_text_file(path)` from `tokenizer/opened_files.py`, which returns **one string per line**, with leading/trailing whitespace stripped.

If `instructions` is passed explicitly, that list is used as-is.

Each non-empty line is expected to represent **one merge rule** for the inner loop (see below).

---

## Step 1 — Character split

```python
splited_word = [word[i] for i in range(len(word))]
```

**Precise meaning:** `word` is treated as a sequence of Unicode code points in Python’s string model. Each `word[i]` is a string of length 1 (one character), including spaces and punctuation as their own cells.

There is **no** pre-tokenization by whitespace at this stage inside `encoder_`.

---

## Step 2 — Apply every instruction line in file order

**Outer loop:** `for instruction in instructions:`

1. `encoded_word = instruction.split()` → a list of whitespace-separated pieces.

2. **Guard:** If `len(encoded_word) != 2`, the line is **skipped entirely** (no merge attempted for that line).

3. Otherwise, treat `encoded_word[0]` as **left** and `encoded_word[1]` as **right**. The merged symbol is always **`left + right`** (string concatenation), not a third stored form.

**Inner loop:** `index` starts at `0` each time a new `instruction` begins.

While `index < len(splited_word) - 1`:

- If `splited_word[index] == left` **and** `splited_word[index+1] == right`:
  - Set `splited_word[index] = left + right`
  - `del splited_word[index + 1]`
  - **Do not** increment `index` (the same index is checked again against the new neighbor to allow chained merges with the same rule in one pass).

- Else:
  - `index += 1`

**Important:** This order is **not** identical to every BPE implementation in the wild, but it is **deterministic** given a fixed `instructions` list and input string.

---

## Step 3 — Return

Return `splited_word` after **all** instruction lines have been applied in sequence.

---

## Example shape (illustrative only)

```text
Input string:  "cat"
After all merges: maybe still ["c","a","t"] if no rule merges those pairs
Or: ["ca","t"] if a rule merged "c" and "a", etc.
```

The exact output depends entirely on your trained `instructions.txt`.

---

## Relationship to training

Training produced `instructions` by repeatedly merging the **training corpus** of words. `encoder_` applies the **same ordered list** to **any new string** characterized as above. That is how inference stays aligned with training.
