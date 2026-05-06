# Overview and data flow

## Three separate pipelines

The project actually has **three** different pipelines. Keeping them separate avoids confusion.

### A. BPE training (offline, Phase 1)

**Goal:** Learn which pairs of symbols to merge repeatedly across a **fixed list of training strings** (usually words), and record (1) the ordered list of merges and (2) a token → integer id table.

**Input:** A JSON file whose values include lists of strings (for example `data_set` or `words` keys).

**Output:**

- A list of merge operations (saved as lines in `instructions.txt`).
- A vocabulary object (saved as JSON with a `tokens_generated` map).

**Code:** `tokenizer/byte_pair_encoding.py`, `tokenizer/normalizer.py`, orchestrated from `main.py` via `run_bpe_training()`.

---

### B. Tokenization at inference (Phase 2)

**Goal:** Turn **arbitrary new text** into integer ids the model could consume, and (for debugging) turn ids back into text.

**Input:** A string plus the **already trained** merge list and vocabulary (from disk or passed in explicitly).

**Output:** Lists of subword strings, then lists of integers, then round-trip strings.

**Code:** `tokenizer/encoder.py`.

**Important:** This path does **not** change the vocabulary. It only **applies** rules produced in pipeline A.

---

### C. Training data preparation (Phase 3)

**Goal:** Build a **long sequence of token ids** from many corpus strings, optionally mark boundaries with special tokens, then cut that stream into **batched tensors** `(X, Y)` for next-token prediction.

**Input:** Many short strings (from JSON and/or text files) plus the same merge list and vocabulary as in B (extended with specials).

**Output:** NumPy arrays for batched language-model windows.

**Code:** `training/data_prep.py`.

---

## One-line dependency chain

```text
JSON words  →  [A] BPE training  →  instructions + vocab JSON
                                            ↓
New text  →  [B] encoder / decoder  →  ids ↔ text
                                            ↓
Many strings  →  [C] data_prep  →  long id stream  →  (X, Y) batches
```

## What runs when

| You run | Pipeline |
|---------|----------|
| `python main.py train_bpe` | A |
| `python main.py` (default) | B (smoke print) |
| `python -m training.data_prep` | C (smoke, using `.env` paths) |
| `python -m training.train` | C then a tiny embedding demo |

This separation is deliberate: **training the tokenizer** is not the same as **encoding a sentence** or **building LM batches**.
