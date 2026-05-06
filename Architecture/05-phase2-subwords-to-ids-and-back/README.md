# Phase 2 — Subwords to ids and back

Source file: `tokenizer/encoder.py`

This document separates three small functions so their roles do not blur together.

---

## 1. `decode_tokens(encoded_tokens, vocab=None, unk_id=None) -> list[int]`

**Input:**

- `encoded_tokens`: list of strings (typically the output of `encoder_`).
- `vocab`: optional dict with key `"tokens_generated"` mapping **string → int**. If `None`, loads from `os.getenv("FILE_GENERATION")` via `open_reads_json_file`.
- `unk_id`: optional int.

**Per-token logic:**

1. `tid = vocab["tokens_generated"].get(token)` (using `.get`).

2. If `tid is None`:
   - If `unk_id is None`, raises **`KeyError`** with the missing token in the message.
   - Otherwise appends `unk_id`.

3. If `tid` is not `None`, appends that integer.

**Output:** A flat Python list of integers, same length as `encoded_tokens` (unless an exception is raised).

**Important:** This function does **not** concatenate strings. It only maps each piece to an id.

---

## 2. `decoder(token_ids, vocab=None) -> str`

**Input:**

- `token_ids`: any iterable of values convertible with `int(...)`.
- `vocab`: same optional resolution as above.

**Internal step — reverse map:**

`_id_to_token` builds `rev: dict[int, str]` by iterating every `(tok, i)` in `tokens_generated` and doing `rev[i] = tok`. If two different strings mapped to the same id (should not happen in a consistent vocab), **later entries overwrite earlier ones** in `rev`.

**Per-id logic:**

For each id `i`, append `rev.get(int(i), "")` to a list of parts. Missing ids become **empty string** pieces (no exception).

**Output:** `"".join(parts)` — **no spaces** inserted between subwords. That matches standard BPE decoding: reconstruction is string concatenation.

---

## 3. `encode_to_ids(text, *, instructions=None, vocab=None, unk_id=None) -> list[int]`

**Definition in code:**

1. `toks = encoder_(text, instructions=instructions)`
2. `return decode_tokens(toks, vocab=vocab, unk_id=unk_id)`

So this is the **composition** of character splitting + merge application + vocabulary lookup.

---

## Round-trip condition (exact)

For a fixed `(instructions, vocab)` pair, a **round trip** means:

```text
text  ==  decoder(encode_to_ids(text, instructions=..., vocab=...), vocab=...)
```

**When it can fail in practice:**

- If `encode_to_ids` produces a token string **missing** from `tokens_generated` and `unk_id` is `None`, `decode_tokens` raises before `decoder` runs.
- If you use `unk_id`, the round-trip string will **not** equal the original text unless you replace UNK handling in `decoder` (currently `decoder` does not map UNK back to original characters).

The unit tests in `tests/test_tokenizer.py` assert the strict round-trip for in-vocabulary paths.

---

## Summary table

| Function | Input type | Output type |
|----------|------------|-------------|
| `encoder_` | `str` | `list[str]` |
| `decode_tokens` | `list[str]` | `list[int]` |
| `decoder` | iterable of ints | `str` |
| `encode_to_ids` | `str` | `list[int]` |
