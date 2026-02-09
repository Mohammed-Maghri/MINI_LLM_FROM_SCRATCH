import os
from pprint import pprint
from dotenv import load_dotenv
from tokenizer.opened_files import (
    open_reads_json_file,
    write_into_json,
    read_text_file)
from tokenizer.byte_pair_encoding import ( 
    start_word_set_generation,
    recur_func )
from tokenizer.encoder import (encoder_ , decode_tokens)
from tokenizer.normalizer import generate_tokens_list
import numpy as np


load_dotenv()


tokens_saved : list[str] = []
tokens_instruction : list[str] = []

# opened_data = open_reads_json_file(os.getenv('DATA_SET_FILE'))
# word_sets = start_word_set_generation(opened_data)
# recur_func(tokens_saved, tokens_instruction, word_sets, 0, 3000)
# write_into_json(os.getenv('FILE_GENERATION'), generate_tokens_list(tokens_saved))
# with open(os.getenv('INSTRUCTIONS_FILE'), 'w', encoding="utf-8") as f:
#     for instruction in tokens_instruction :
#         f.write(instruction + "\n")


vocab_set = open_reads_json_file(os.getenv('FILE_GENERATION'))
encoded_token = encoder_("he enjoys hiking", read_text_file(os.getenv('INSTRUCTIONS_FILE')))
decode_tok = decode_tokens(encoded_token, vocab_set)

print(np.random.randn(len(vocab_set.get("tokens_generated", {})), 64))
# print(len(vocab_set.get("tokens_generated", {})))


print(f"encoded_token : {encoded_token}")
print(f"decode_tok : {decode_tok}")