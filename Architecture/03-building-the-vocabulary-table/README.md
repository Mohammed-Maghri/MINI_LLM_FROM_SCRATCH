# Building the vocabulary table (token → id)

Source file: `tokenizer/normalizer.py`  
Called from `main.py` **after** `recur_func` finishes: `vocab = generate_tokens_list(tokens_saved)`.

---

## What `tokens_saved` is at this point

A list of Python strings, in **merge order**, produced during BPE training. Each string is one merged token vocabulary entry (for example `"th"`, `"ing"`, … depending on your data and merge cap).

The **order in this list matters**: the **first** merge gets id `256`, the second gets `257`, and so on.

---

## Function: `generate_tokens_list(tokens: list[str]) -> dict`

**Return value shape:**

```python
{"tokens_generated": { "<token_string>": <integer_id>, ... }}
```

### Part A — Byte rows (ids `0` through `254`)

**Code:**

```python
for i in range(255):
    generated_tokens["tokens_generated"][bytes([i]).decode("latin-1")] = i
```

**Precise meaning:**

- `i` runs `0, 1, …, 254` (the loop is `range(255)`, so **255 is not included**).
- For each `i`, the key is the single-character string obtained by treating `i` as a byte value and decoding with **Latin-1**.
- The value stored is `i`.

So you get one table entry per byte value `0`…`254` as a one-character token.

**Note:** In this implementation, byte value **255** is **not** inserted by this loop. That is a literal property of `range(255)`; if you need full `0…255` coverage, the loop would need to change to `range(256)`.

### Part B — Merge rows (ids starting at `256`)

**Code:**

```python
for i in range(len(tokens)):
    generated_tokens["tokens_generated"][tokens[i]] = i + 256
```

**Precise meaning:**

- `tokens` is the same list as `tokens_saved` from training (merge strings in order).
- The **first** merge string `tokens[0]` receives id **`256`**.
- The **second** receives **`257`**, etc.

**Collision rule:** If a merge string equals a byte string that was already inserted in Part A, the later assignment **overwrites** the earlier id for that key. In normal BPE training the merge strings are usually longer than one character, so collisions are rare, but the dictionary semantics are overwrite-on-duplicate-key.

---

## After this function returns

`main.py` writes the dict to disk:

1. `write_into_json(os.getenv("FILE_GENERATION"), vocab)` — typically `tokenized_data/output_set.json`.
2. Each line of `tokens_instruction` is written to `INSTRUCTIONS_FILE`.
3. `save_tokenizer` also writes `VOCAB_JSON` (often `tokenized_data/vocab.json`) with the same vocab dict.

From here on, **every** encode/decode step looks up strings in `tokens_generated` and applies merges in `instructions.txt` order.

---

## Mental model

| Id range | Meaning in this codebase |
|----------|---------------------------|
| `0 …` (loop-dependent) | Single-byte character tokens (Latin-1 decode) |
| `256 … 256 + len(merges) - 1` | Merge tokens, **in the same order** merges were chosen during training |

The merge **lines** on disk do **not** contain ids; ids are **only** in the JSON map.
