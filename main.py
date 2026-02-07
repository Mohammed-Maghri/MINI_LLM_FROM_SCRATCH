import os
from pprint import pprint
from dotenv import load_dotenv
from functions.opened_files import (open_reads_json_file, write_into_json)
from functions.byte_pair_encoding import ( 
    start_word_set_generation,
    recur_func )

load_dotenv()

def generate_tokens_list (token_saved : list[str], tokens : list[str]) -> dict :
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

opened_data = open_reads_json_file(os.getenv('DATA_SET_FILE'))
word_sets = start_word_set_generation(opened_data)
recur_func(tokens_saved,word_sets, 0, 1000)
# print(f"!!!! -- {tokens_saved}")
write_into_json(os.getenv('FILE_GENERATION'), generate_tokens_list(tokens_saved, tokens_saved))
