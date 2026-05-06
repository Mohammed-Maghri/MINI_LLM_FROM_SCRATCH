# Building an LLM from Scratch

A comprehensive implementation of a Large Language Model built entirely from the ground up to understand the fundamental components and mechanisms that power modern AI systems.

## Overview

This project aims to demystify Large Language Models by implementing each component from scratch, without relying on existing ML frameworks. Rather than using pre-built libraries, we build every piece of the pipeline to gain deep insight into how models like GPT actually work under the hood.

## Project Status

**Current Phase:** Phase 3 complete — ready for **Phase 4: Embedding Layer**  
**Repository root:** `MINI_LLM_FROM_SCRATCH` (all paths below are relative to this directory)

### Progress Tracker

- [x] **Phase 1: Tokenization Training**
  - [x] Byte Pair Encoding (BPE) algorithm implementation
  - [x] Training data preparation (morphological patterns in `data_set/data.json`)
  - [x] Frequency-based list (`data_set/common_words_1000.json`)
  - [x] Vocabulary generation (size grows with merge count; byte base 0–254 plus learned merges)
  - [x] Merge cap via `BPE_MAX_MERGES` in `.env`
  - [x] Scaling notes for larger corpora (streaming, parallel counting)
- [x] **Phase 2: Complete Tokenizer**
  - [x] `encoder_` — text → BPE subword strings
  - [x] `decode_tokens` — subwords → numeric ids
  - [x] `decoder` — ids → text (round-trip with `encode_to_ids`)
  - [x] Save/load: `tokenized_data/instructions.txt` + `tokenized_data/vocab.json` (and `output_set.json` for backward compatibility)
  - [x] Automated round-trip tests in `tests/test_tokenizer.py`
- [x] **Phase 3: Data Preparation**
  - [x] Collect training corpus (`training/data_prep.collect_corpus` from JSON + optional `corpus/*.txt`)
  - [x] Tokenize corpus to a flat id stream (`tokenize_corpus` + `encode_to_ids`)
  - [x] LM batches (`make_lm_batches`) for next-token prediction
  - [x] Special tokens `<PAD>`, `<BOS>`, `<EOS>`, `<UNK>` (`tokenizer/special_tokens.py`)
- [ ] **Phase 4: Embedding Layer**
  - [ ] Token embeddings
  - [ ] Positional encodings
  - [ ] Embedding combination
- [ ] **Phase 5: Transformer Architecture**
  - [ ] Self-attention mechanism
  - [ ] Multi-head attention
  - [ ] Feed-forward networks
  - [ ] Layer normalization
  - [ ] Residual connections
  - [ ] Stack transformer blocks
- [ ] **Phase 6: Training Infrastructure**
  - [ ] Loss function (cross-entropy)
  - [ ] Optimizer (Adam)
  - [ ] Training loop
  - [ ] Checkpointing
  - [ ] Metrics tracking
- [ ] **Phase 7: Text Generation**
  - [ ] Autoregressive sampling
  - [ ] Temperature scaling
  - [ ] Top-k sampling
  - [ ] Top-p (nucleus) sampling
- [ ] **Phase 8: Evaluation**
  - [ ] Perplexity calculation
  - [ ] Qualitative testing
  - [ ] Hyperparameter tuning

## What We've Built So Far

**Completed Phases:**

- ✅ Phase 1: BPE training (`tokenizer/byte_pair_encoding.py`)
- ✅ Phase 2: Encoder / decoder API (`tokenizer/encoder.py`) — round-trip verified in tests
- ✅ Phase 3: Corpus → ids → batches (`training/data_prep.py`)

### Phase 1: Byte Pair Encoding (BPE) Tokenizer

Successfully implemented the tokenization algorithm used by GPT-2, GPT-3, and GPT-4 to convert text into numerical tokens.

#### Algorithm Overview

BPE works by iteratively merging the most frequent adjacent character pairs:

```
1. Start: "playing" → ['p','l','a','y','i','n','g']
2. Merge most frequent pair → ['p','l','ay','i','n','g']
3. Continue merging → ['p','lay','i','n','g']
4. Build vocabulary → ['play','i','n','g']
```

#### Implementation Details

**Core Functions:**

- `start_word_set_generation()` - Initializes character-level splits
- `sliding_window_comparison()` - Finds most frequent pairs
- `replace_voted_token()` - Applies merge operations
- `recur_func()` - Orchestrates iterative training
- `generate_tokens_list()` - Creates final vocabulary

**Training Datasets:**

1. **Initial Dataset (`data.json`):**
   - 1000 words with morphological variations
   - Common prefixes (un-, re-, dis-, pre-)
   - Common suffixes (-ing, -ed, -er, -ly, -ness, -able)
   - Generated vocabulary size = number of merge steps plus byte rows (see `output_set.json`)

2. **Frequency-Based Dataset (`common_words_1000.json`):** ✨ NEW
   - 1000 most commonly used English words
   - Based on corpus frequency analysis
   - Includes core function words, common verbs, nouns, adjectives
   - Optimized for real-world language patterns

**Generated vocabulary (shape):**

- Base: single-byte tokens (Latin-1) for ids starting at 0 (see `tokenizer/normalizer.generate_tokens_list`)
- Learned: each BPE merge adds one token string mapped to the next integer
- Total size: `len(tokens_generated)` in `tokenized_data/output_set.json` or `vocab.json` (depends on `BPE_MAX_MERGES` and training data)

**Sample learned tokens** (exact ids vary after retraining):

```
"in", "ing", "er", "ed", "ness", "play", "correct", "understand", …
```

**Dataset Comparison:**

| Dataset                     | Purpose                           | Best For                      |
| --------------------------- | --------------------------------- | ----------------------------- |
| `data.json`                 | Morphological patterns            | Learning grammatical suffixes |
| `common_words_1000.json` ✨ | Real-world frequency distribution | Production-like tokenization  |

The frequency-based dataset provides more realistic token distributions matching actual English usage, while the morphological dataset is better for understanding how BPE learns linguistic structures.

### Phase 2: Encoder, decoder, persistence

The pipeline splits into **merge application** (strings) and **vocabulary lookup** (ids), then **decode** concatenates pieces back into text (standard BPE decode).

| Step | Function | Role |
|------|----------|------|
| Text → subwords | `encoder_(text, instructions=None)` | Greedy application of merge lines from `instructions.txt` |
| Subwords → ids | `decode_tokens(tokens, vocab=None, unk_id=None)` | Looks up each piece in `tokens_generated` |
| One-shot text → ids | `encode_to_ids(...)` | `encoder_` then `decode_tokens` |
| Ids → text | `decoder(token_ids, vocab=None)` | Inverse map id → string, then concatenate |

#### Files and outputs

| Artifact | Purpose |
|----------|---------|
| `tokenized_data/instructions.txt` | One merge rule per line: `left right` (two symbols merged into `left+right`) |
| `tokenized_data/output_set.json` | `{"tokens_generated": { "<token>": <id>, ... } }` — primary vocab for `.env` `FILE_GENERATION` |
| `tokenized_data/vocab.json` | Same JSON schema; written beside instructions when you run `python main.py train_bpe` (or copy manually) |

Programmatic save/load helpers live in `tokenizer/tokenizer_bundle.py` (`save_tokenizer`, `load_raw_vocab`, `load_instructions`).

#### Round-trip check

```text
text  --encode_to_ids-->  [id, ...]  --decoder-->  text
```

Verified by `tests/test_tokenizer.py` (run from repo root with `python -m unittest tests.test_tokenizer -v`).

## Project structure

```text
MINI_LLM_FROM_SCRATCH/
├── main.py                     # BPE training (`train_bpe`) + encode/decode smoke
├── tokenizer/
│   ├── byte_pair_encoding.py   # BPE training loop
│   ├── encoder.py              # encoder_, decode_tokens, decoder, encode_to_ids
│   ├── tokenizer_bundle.py   # save/load instructions + vocab JSON
│   ├── special_tokens.py       # <PAD>, <BOS>, <EOS>, <UNK> extension
│   ├── opened_files.py         # JSON helpers used by training scripts
│   ├── normalizer.py           # `generate_tokens_list` (byte vocab + merges)
│   └── data_set_loader.py      # Optional PDF ingestion helpers
├── training/
│   ├── data_prep.py            # Phase 3: corpus, tokenize, batches
│   └── train.py                # Tiny demo: batches + random embedding matrix
├── tests/
│   └── test_tokenizer.py       # Round-trip + UNK tests
├── data_set/
│   ├── data.json               # Large word list (e.g. book-derived tokens)
│   └── common_words_1000.json  # 1000 high-frequency words
├── tokenized_data/
│   ├── instructions.txt        # Merge rules (generated)
│   ├── output_set.json         # Vocab (generated; `FILE_GENERATION`)
│   └── vocab.json              # Same as vocab sidecar (`VOCAB_JSON`)
├── .env
└── README.md
```

## How BPE Tokenization Works

### Training Phase (Completed)

1. **Initialize:** Split all words into individual characters
2. **Count:** Find frequency of all adjacent character pairs
3. **Merge:** Combine the most frequent pair into a single token
4. **Repeat:** Continue until vocabulary size reached or frequency threshold met
5. **Save:** Store learned merge rules and vocabulary mappings

### Encoding (inference)

1. Split the string into Unicode characters (one char per cell).
2. Walk `instructions.txt` in order; each line `a b` replaces adjacent `a` and `b` with the merged symbol `ab` until no rule applies for that line.
3. Map each resulting subword string to an integer with `tokens_generated`.

Use `encode_to_ids` for steps 2–3 in one call when `instructions` and `vocab` are loaded (or rely on `.env` paths).

### Decoding (inference)

1. Map each id to its token string (reverse of `tokens_generated`).
2. Concatenate strings with no extra spaces (BPE-style).

Use `decoder` for the full sequence.

### Why BPE?

- **Balances vocabulary size and sequence length**
- **Handles rare/unknown words** through subword decomposition
- **Captures morphology** (prefixes, suffixes, roots)
- **Language-agnostic** - purely statistical approach
- **Same algorithm used by GPT-2/3/4, RoBERTa, BART**

## Phase 3: Data preparation (implemented)

Module: `training/data_prep.py`.

1. **`collect_corpus(json_paths=..., text_glob=...)`**  
   Loads strings from JSON files under `data_set/` (`data_set`, `words`, or a bare JSON list) and optionally every non-empty line from text files matching a glob (for example `corpus/*.txt` once you add files there).

2. **`tokenize_corpus(...)`**  
   Runs `encode_to_ids` per segment, inserts `<EOS>` between segments by default, and can prepend `<BOS>` per segment with `bos_every=True`. Unknown subwords resolve to `<UNK>` when you pass the extended vocabulary from step 4.

3. **`make_lm_batches(token_ids, seq_len, batch_size)`**  
   Returns NumPy arrays `X` and `Y` with shape `(num_batches, batch_size, seq_len)` where each row is a contiguous slice of the corpus and `Y` is the next-token target (shifted by one inside the slice).

4. **Special tokens**  
   `extend_vocab_with_specials` in `tokenizer/special_tokens.py` appends `<PAD>`, `<BOS>`, `<EOS>`, `<UNK>` **after** the current maximum id so you do not collide with byte or merge ids. Training embeddings should use `len(tokens_generated)` after extension.

Convenience entry point: **`prepare_training_bundle(...)`** wires corpus → specials → ids → batches.

Smoke test (uses `PHASE3_SMOKE_JSON` in `.env`, default `common_words_1000.json`, so the huge `data.json` is not loaded by accident):

```bash
python -m training.data_prep
```

### Next: build the model (Phase 4 onward)

1. **Embeddings:** Convert token IDs to dense vectors
2. **Positional Encoding:** Add position information
3. **Transformer Layers:** Self-attention + feed-forward
4. **Output Layer:** Project to vocabulary size
5. **Training:** Optimize on large text corpus
6. **Generation:** Sample text autoregressively

## Key Learning Outcomes

### Understanding Gained

- How tokenizers compress text efficiently
- Why subword units are superior to characters or words
- Role of frequency thresholds in preventing overfitting
- Importance of merge order in building complex patterns
- How production LLMs (GPT) handle tokenization at scale
- **Complete tokenization pipeline:** Text ↔ Token IDs bidirectional conversion
- **Round-trip verification:** Ensuring encoding/decoding are inverse operations
- **Vocabulary management:** Mapping between tokens, IDs, and text

### Technical Skills

- Implementing greedy algorithms
- Statistical pattern recognition
- Vocabulary design and optimization
- Recursive data structure manipulation
- Text preprocessing for ML
- **Bidirectional mappings:** Creating and using token↔ID dictionaries
- **Sequential rule application:** Applying ordered transformations
- **Data pipeline design:** Chaining encode/decode operations

## Technical Specifications

**Language:** Python 3.x  
**Dependencies:**

- `json` - Data serialization
- `collections.Counter` - Frequency counting
- `python-dotenv` - Environment configuration

**Performance:**

- Training: ~1-2 seconds on 1000 words
- Encoding: ~0.01 seconds per 1000 characters ✓ Implemented
- Decoding: Instant (dictionary lookup) ✓ Implemented
- Vocabulary size: see `len(tokens_generated)` after training
- Compression ratio: depends on text (typically fewer tokens than characters for English)
- Merge rules: one line per merge in `instructions.txt`

## Comparison to Production Systems

### Our Implementation vs GPT-2

| Feature        | Our Implementation | GPT-2            |
| -------------- | ------------------ | ---------------- |
| Algorithm      | BPE                | BPE              |
| Base Vocab     | 256 bytes          | 256 bytes        |
| Learned Tokens | (BPE merge count)  | 50,000           |
| Total Vocab    | configurable       | 50,257           |
| Training Data  | 1000 words         | 40GB text        |
| Encoding Speed | ~0.01s/1K chars    | ~0.003s/1K chars |

**Key Insight:** The algorithm is identical - only scale differs!

## Future Enhancements

### Tokenizer Improvements

- Migrate to byte-level BPE (handle any Unicode)
- Add pre-tokenization (split on whitespace/punctuation)
- Implement vocabulary pruning
- Add caching for common words
- Optimize with trie data structures

### Large-Scale Training Optimizations

For handling massive datasets (millions+ words):

- **Memory Management:** Streaming/batching instead of loading all data
- **Incremental Counting:** Update pair frequencies without storing all pairs
- **Parallel Processing:** Multi-core pair counting and merging
- **Early Stopping:** Halt when merge frequency drops below threshold
- **Efficient Data Structures:** NumPy arrays for token sequences
- **Caching:** Store and reuse intermediate merge results

### Model Improvements

- Implement attention visualization
- Add model parallelism for scaling
- Implement gradient checkpointing
- Add mixed precision training
- Implement beam search for generation

## Resources Referenced

### Papers

- "Attention Is All You Need" (Vaswani et al., 2017) - Transformer architecture
- "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019) - GPT-2
- "Neural Machine Translation of Rare Words with Subword Units" (Sennrich et al., 2016) - BPE

### Concepts

- Byte Pair Encoding
- Subword tokenization
- Transformer architecture
- Self-attention mechanisms
- Autoregressive language modeling

## Installation and usage

### Environment (recommended)

On PEP 668–managed Python, use a virtual environment:

```bash
cd /path/to/MINI_LLM_FROM_SCRATCH
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### Train or refresh the tokenizer

```bash
.venv/bin/python main.py train_bpe
```

This reads `DATA_SET_FILE` and `BPE_MAX_MERGES` from `.env`, writes `FILE_GENERATION` (JSON vocab), `INSTRUCTIONS_FILE`, and `VOCAB_JSON`.

Default `.env` points at `data_set/common_words_1000.json` for a faster training loop; switch to `data_set/data.json` for a larger word list.

### Day-to-day: encode and decode

```bash
.venv/bin/python main.py
```

prints a short encode/decode sample using `ENCODE_SAMPLE` (optional in `.env`).

### Programmatic API

```python
from tokenizer.encoder import encoder_, decode_tokens, decoder, encode_to_ids
from tokenizer.opened_files import open_reads_json_file
from tokenizer.tokenizer_bundle import load_instructions

vocab = open_reads_json_file("tokenized_data/output_set.json")
instructions = load_instructions("tokenized_data/instructions.txt")

text = "journey"
assert decoder(encode_to_ids(text, instructions=instructions, vocab=vocab), vocab=vocab) == text
```

### Tests

```bash
.venv/bin/python -m unittest tests.test_tokenizer -v
```

### Training data demo (batches)

```bash
.venv/bin/python -m training.train
```

## Contributing

This is an educational project focused on understanding LLM internals. The goal is learning, not production deployment.

## License

This project is for educational purposes.

## Acknowledgments

Built as a deep dive into LLM architecture and training, demonstrating that complex AI systems can be understood and implemented with foundational knowledge and systematic engineering.

---

**Status:** Phases 1–3 complete (tokenizer + data prep)  
**Next:** Phase 4 — embedding layer  
**Last Updated:** May 6, 2026
