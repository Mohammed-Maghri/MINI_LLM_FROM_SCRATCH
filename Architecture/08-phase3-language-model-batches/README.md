# Phase 3 — Language-model batches (`make_lm_batches`)

Source file: `training/data_prep.py`  
Depends on: NumPy.

---

## Input

- `token_ids: list[int]` — the long stream produced by `tokenize_corpus`.
- `seq_len: int` — context length for **each** input row inside a batch (must be ≥ 2).
- `batch_size: int` — how many parallel rows are grouped into one batch slice.

---

## Step 1 — Convert to a 1-D NumPy array

```python
ids = np.asarray(token_ids, dtype=np.int64)
```

Early errors:

- If `len(ids) <= seq_len`, raise (not enough tokens for even one window).
- If `seq_len < 2`, raise.

---

## Step 2 — Build non-overlapping contiguous windows

**Loop variable:** `i` runs `0, seq_len, 2*seq_len, …` via `range(0, len(ids) - seq_len, seq_len)` (non-overlapping blocks of length `seq_len`, stepping by `seq_len`).

**For each `i`:**

1. `chunk = ids[i : i + seq_len + 1]` — length should be `seq_len + 1` to split into `x` and `y`.

2. If `len(chunk) < seq_len + 1`, **break** the loop (end of array not long enough for another full chunk).

3. Append `chunk[:-1]` to `x_rows` (length `seq_len`).

4. Append `chunk[1:]` to `y_rows` (length `seq_len`).

**Relationship inside one row:**

For row `r` with chunk starting at position `i`:

- `x_rows[r][t] == ids[i + t]` for `t = 0 … seq_len - 1`
- `y_rows[r][t] == ids[i + t + 1]` for `t = 0 … seq_len - 1`

So for all `t` from `0` to `seq_len - 2`, you have **`y[t] == x[t+1]`**. For `t == seq_len - 1`, **`y[seq_len-1]`** is **`ids[i + seq_len]`**, which is the token **immediately after** the last token of `x` in the original stream. That is the standard “predict next token at each position” alignment for a fixed-length window.

**Between rows:** The next row starts at `i + seq_len`, so the previous row used ids from `i` through `i + seq_len` inclusive for targets. The next row’s `x` starts at `ids[i + seq_len]`, which equals the **last** `y` token of the previous row. So rows are **staggered by one position** across chunk boundaries: the stream is continuous in the autoregressive sense at chunk edges.

---

## Step 3 — Stack rows into matrices

```python
x_stack = np.stack(x_rows, axis=0)   # shape (num_rows, seq_len)
y_stack = np.stack(y_rows, axis=0)
```

If no rows were collected, raise.

---

## Step 4 — Trim to a multiple of `batch_size`

```python
n = (x_stack.shape[0] // batch_size) * batch_size
x_stack = x_stack[:n]
y_stack = y_stack[:n]
```

If `n == 0`, raise (“Not enough rows for batch_size”).

---

## Step 5 — Reshape to `(num_batches, batch_size, seq_len)`

```python
new_shape = (n // batch_size, batch_size, seq_len)
return x_stack.reshape(new_shape), y_stack.reshape(new_shape)
```

**Final shapes:**

- `X` has shape `(num_batches, batch_size, seq_len)`.
- `Y` has the **same** shape.

**Indexing meaning:** After reshape, `X[b, r, t]` is the **t**-th token in training row **r** inside minibatch **b** (axis `0` = minibatch index along the corpus, axis `1` = row within that minibatch, axis `2` = time).

---

## `prepare_training_bundle` (wires everything)

Order of operations:

1. `extend_vocab_with_specials(base_vocab)` → extended vocab + `SpecialTokenIds`.

2. `collect_corpus(...)` → list of strings.

3. `tokenize_corpus(...)` → long id list.

4. `make_lm_batches(ids, seq_len, batch_size)` → `(X, Y)`.

Returns `(X, Y, vocab_ext, specials)` so downstream code knows embedding row count and special id values.
