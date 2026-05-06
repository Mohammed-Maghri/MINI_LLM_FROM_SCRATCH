import os
import unittest
from pathlib import Path

from dotenv import load_dotenv

from tokenizer.encoder import decode_tokens, decoder, encoder_, encode_to_ids
from tokenizer.opened_files import open_reads_json_file
from tokenizer.special_tokens import extend_vocab_with_specials
from tokenizer.tokenizer_bundle import load_instructions


ROOT = Path(__file__).resolve().parents[1]


class TestTokenizerRoundTrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_dotenv(ROOT / ".env")
        os.chdir(ROOT)
        cls.vocab_path = ROOT / os.getenv("FILE_GENERATION", "tokenized_data/output_set.json")
        cls.instr_path = ROOT / os.getenv("INSTRUCTIONS_FILE", "tokenized_data/instructions.txt")
        cls.vocab = open_reads_json_file(str(cls.vocab_path))
        cls.instructions = load_instructions(str(cls.instr_path))

    def test_encoder_returns_strings(self):
        toks = encoder_("journey", instructions=self.instructions)
        self.assertIsInstance(toks, list)
        self.assertTrue(all(isinstance(t, str) for t in toks))

    def test_round_trip_ascii_word(self):
        text = "journey"
        toks = encoder_(text, instructions=self.instructions)
        ids = decode_tokens(toks, vocab=self.vocab)
        out = decoder(ids, vocab=self.vocab)
        self.assertEqual(out, text)

    def test_encode_to_ids_matches_pipeline(self):
        text = "the"
        a = encode_to_ids(text, instructions=self.instructions, vocab=self.vocab)
        b = decode_tokens(encoder_(text, instructions=self.instructions), vocab=self.vocab)
        self.assertEqual(a, b)

    def test_extended_vocab_unk(self):
        vocab_ext, sp = extend_vocab_with_specials(self.vocab)
        fake = ["<not-a-real-token>"]
        ids = decode_tokens(fake, vocab=vocab_ext, unk_id=sp.unk)
        self.assertEqual(ids, [sp.unk])


if __name__ == "__main__":
    unittest.main()
