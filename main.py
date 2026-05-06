import os

from dotenv import load_dotenv

from tokenizer.byte_pair_encoding import recur_func, start_word_set_generation
from tokenizer.encoder import decoder, encode_to_ids
from tokenizer.normalizer import generate_tokens_list
from tokenizer.opened_files import open_reads_json_file, write_into_json
from tokenizer.tokenizer_bundle import save_tokenizer

load_dotenv()

tokens_saved: list[str] = []
tokens_instruction: list[str] = []


def run_bpe_training() -> None:
    """Train BPE from DATA_SET_FILE and write vocabulary + merge rules + vocab.json."""
    opened_data = open_reads_json_file(os.getenv("DATA_SET_FILE", "data_set/data.json"))
    word_sets = start_word_set_generation(opened_data)
    max_merges = int(os.getenv("BPE_MAX_MERGES", "3000"))
    recur_func(tokens_saved, tokens_instruction, word_sets, 0, max_merges)
    vocab = generate_tokens_list(tokens_saved)
    out_json = os.getenv("FILE_GENERATION", "tokenized_data/output_set.json")
    write_into_json(out_json, vocab)
    instr_path = os.getenv("INSTRUCTIONS_FILE", "tokenized_data/instructions.txt")
    with open(instr_path, "w", encoding="utf-8") as f:
        for instruction in tokens_instruction:
            f.write(instruction + "\n")
    vocab_json = os.getenv("VOCAB_JSON", "tokenized_data/vocab.json")
    save_tokenizer(vocab, tokens_instruction, vocab_json_path=vocab_json, instructions_path=instr_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "train_bpe":
        run_bpe_training()
        print("Wrote tokenizer to", os.getenv("FILE_GENERATION"), os.getenv("INSTRUCTIONS_FILE"), os.getenv("VOCAB_JSON"))
        sys.exit(0)

    sample = os.getenv("ENCODE_SAMPLE", "journey")
    ids = encode_to_ids(sample)
    back = decoder(ids)
    print("Encode/decode sample:", repr(sample), "->", ids, "->", repr(back))
    print("Run: python main.py train_bpe   # retrain tokenizer")
    print("Run: python -m training.data_prep   # Phase 3 batching smoke test")
    print("Run: python -m training.train   # batches + random embedding rows")
