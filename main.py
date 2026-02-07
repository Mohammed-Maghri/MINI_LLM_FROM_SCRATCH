import os
from pprint import pprint
from dotenv import load_dotenv
from functions.opened_files import (
    open_reads_json_file,
    write_into_json,
    read_text_file)
from functions.byte_pair_encoding import ( 
    start_word_set_generation,
    recur_func )

load_dotenv()

def generate_tokens_list (tokens : list[str]) -> dict :
    generated_tokens : dict = {"tokens_generated" : []}
    try : 
        for i in range(255) :
            generated_tokens.get("tokens_generated", []).append([bytes([i]).decode("latin-1"), i])
        for i in range(len(tokens)) :
            generated_tokens.get("tokens_generated", []).append([tokens[i], i + 256])
    except Exception as e :
        pprint(f"Error Occured !!{e}")
    return generated_tokens


tokens_saved : list[str] = []
tokens_instruction : list[str] = []

opened_data = open_reads_json_file(os.getenv('DATA_SET_FILE'))
word_sets = start_word_set_generation(opened_data)
recur_func(tokens_saved, tokens_instruction, word_sets, 0, 1000)

# print(f"tokens_saved : {tokens_saved}")
# print(f"tokens_instruction : {tokens_instruction}")
# print(f"!!!! -- {tokens_saved}")

write_into_json(os.getenv('FILE_GENERATION'), generate_tokens_list(tokens_saved))
with open(os.getenv('INSTRUCTIONS_FILE'), 'w', encoding="latin-1") as f:
    for instruction in tokens_instruction :
        f.write(instruction + "\n")

pprint(read_text_file(os.getenv('INSTRUCTIONS_FILE')))
