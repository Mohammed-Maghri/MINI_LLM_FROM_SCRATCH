# Special tokens (`<PAD>`, `<BOS>`, `<EOS>`, `<UNK>`)

Source file: `tokenizer/special_tokens.py`

These tokens exist so a later neural model can pad batches, mark sequence starts/ends, and absorb unknown subwords during **training-data** construction. They are **not** produced by the BPE training loop in `byte_pair_encoding.py`.

---

## String constants

```python
PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"
```

---

## `extend_vocab_with_specials(vocab: dict) -> tuple[dict, SpecialTokenIds]`

**Input:** `vocab` is expected to look like `{"tokens_generated": { ... }}` as produced by `generate_tokens_list` (plus optional other keys ignored here).

**Step 1 — Copy**

```python
inner = copy.deepcopy(vocab.get("tokens_generated", {}))
```

**Step 2 — Find the current maximum id**

```python
start = max(inner.values()) + 1   # or -1 + 1 if empty, yielding 0
```

(Implementation uses helper `_max_token_id`; if the dict is empty, max is `-1`, so `start` becomes `0`.)

**Step 3 — Append four new entries**

Sequential ids, always contiguous:

| Key string | Id value |
|------------|----------|
| `"<PAD>"` | `start` |
| `"<BOS>"` | `start + 1` |
| `"<EOS>"` | `start + 2` |
| `"<UNK>"` | `start + 3` |

**Return:**

1. New dict: `{"tokens_generated": inner}` (only that key in the outer dict).

2. `SpecialTokenIds(pad=start, bos=start+1, eos=start+2, unk=start+3)` — a frozen dataclass for type-safe access.

---

## Why ids are placed **after** the max merge id

BPE already consumed integer ids `0…254` (byte loop) and `256…` for merges. Reserving new symbols **above** the current max guarantees:

- No collision with an existing merge or byte token string key in the dict.
- The embedding matrix row count for the model is simply `len(tokens_generated)` after extension.

---

## Where specials are used

| Location | Usage |
|----------|--------|
| `tokenize_corpus` | Inserts `bos` optionally, appends `eos` between segments, passes `unk` into `encode_to_ids` |
| `training/train.py` | Uses extended vocab size to size a random embedding table demo |

**Not automatically used in:** `decoder` — if you decode raw id streams that contain special ids, you will see their string forms (`"<EOS>"` etc.) concatenated into the output string unless you post-process.

---

## `vocab_size(vocab) -> int`

Returns `len(vocab.get("tokens_generated", {}))` — a convenience for sizing layers after extension.
