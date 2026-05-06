# Phase 1: BPE training (precise steps)

Source file: `tokenizer/byte_pair_encoding.py`  
Orchestration: `main.py` → `run_bpe_training()` loads JSON, then calls the functions below.

---

## Step 1 — Load JSON into per-word character lists

**Function:** `start_word_set_generation(data: dict) -> dict`

**What it does:**

1. Initializes `word_sets = []`.
2. Iterates `for key, values in data.items()`.
3. Whenever `values` is a `list`, iterates each `value` in that list.
4. Treats each `value` as a string and converts it to `list(value)` (a Python list of **one-character strings**).
5. Appends that list to `word_sets`.

**Result shape:** A dict:

```python
{"word_sets": [ ["c","a","t"], ["d","o","g"], ... ], "tokens_size": <copied from data or 0>}
```

So every training item is still **character-split**. No merges have been applied yet.

**JSON contract:** Your training JSON must expose at least one key whose value is a **list of strings**. Examples this repo supports elsewhere: `"data_set"`, `"words"`.

---

## Step 2 — Count adjacent pairs across the whole corpus

**Function:** `sliding_window_comparison(word_sets: dict) -> list[dict]`

**What it does:**

1. Builds `tokens_vote`: for every inner list in `word_sets["word_sets"]`, for every index `i` from `0` to `len(word_set) - 2`, it records the ordered pair `(word_set[i], word_set[i+1])`.

2. Feeds all those pairs into `collections.Counter`.

3. If there are no pairs, returns `[]`.

4. Finds `max_value`, the highest count among all pairs.

5. **Stopping rule:** If `max_value == 1`, returns `[]` (no pair appears more than once globally, so training stops merging).

6. Otherwise, for every pair whose count equals `max_value`, appends a dict:

   - `"token_vocab"`: the two adjacent characters **concatenated** into one string (the merged symbol).
   - `"instruction"`: the same two symbols as a string with a **space** between them (so the line can be split into left and right for inference).

**Tie-breaking:** If several pairs tie for maximum frequency, **all** of them are appended to `all_tokens`. The next step only uses the **first** entry.

---

## Step 3 — Apply one merge everywhere

**Function:** `replace_voted_token(word_sets: dict, voted_token: str) -> dict`

**What it does:**

For each word’s character list:

1. Walks with index `i` from `0` while `i < len(word_correction) - 1`.

2. If `word_correction[i] + word_correction[i+1] == voted_token` (string concatenation equals the chosen merge string), then:
   - Sets `word_correction[i] = voted_token`
   - Removes `word_correction[i+1]` with `pop`
   - **Does not** skip `i`; it stays on the same index so another overlap can be resolved in the same pass.

3. If no match, increments `i` by 1.

4. Replaces `word_sets["word_sets"]` with the new lists.

So one call of this function applies **one** merge symbol **greedily left-to-right** within each word.

---

## Step 4 — Repeat merges until a cap or stop condition

**Function:** `recur_func(tokens_saved, tokens_instruction, word_set, counter, max_tokens) -> None`

**Parameters:**

- `tokens_saved`: list **mutated in place**; each chosen merge’s `token_vocab` string is appended.
- `tokens_instruction`: list **mutated in place**; each chosen merge’s `instruction` string is appended.
- `word_set`: the dict from `start_word_set_generation`, updated in place through merges.
- `counter`: how many merges have been performed so far.
- `max_tokens`: hard stop (from `.env` as `BPE_MAX_MERGES` in `main.py`).

**Logic:**

1. If `counter == max_tokens`, return.

2. Call `sliding_window_comparison(word_set)`.

3. If the result is empty, return.

4. Take `tokens[0]` only: `selected_token` and `token_instruction`.

5. Call `replace_voted_token(word_set, selected_token)`.

6. Append to `tokens_saved` and `tokens_instruction`.

7. Recurse with `counter + 1`.

**Net effect:** You get two parallel lists of length = number of merges actually performed (up to `max_tokens`):

- `tokens_saved[i]` is the merged string introduced at step `i`.
- `tokens_instruction[i]` is the `"a b"` line describing that merge.

Those two lists are what later become the **vocabulary entries for ids ≥ 256** and the **lines in `instructions.txt`**, but that assignment happens in `generate_tokens_list`, not in this file.

---

## Summary table (Phase 1 only)

| Step | Function | Input → output |
|------|----------|----------------|
| 1 | `start_word_set_generation` | JSON dict → `{word_sets: [[chars]...]}` |
| 2 | `sliding_window_comparison` | word_sets → candidate merges for max-frequency pair(s) |
| 3 | `replace_voted_token` | one merge string → updated word_sets |
| 4 | `recur_func` | repeat 2–3 until cap or no mergeable pair |

This is **only** about learning merges from the training word list. It does **not** assign final numeric ids; see the next document.
