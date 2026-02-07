import os
from pprint import pprint
from dotenv import load_dotenv
from functions.opened_files import (open_reads_json_file, write_into_json)
from functions.byte_pair_encoding import ( 
    start_word_set_generation,
    recur_func )

load_dotenv()

def generate_tokens (token_saved : list[str]) -> None :
    try : 
        for i in range(255) :
            pprint(f"{i} -- {bytes([i]).decode("latin-1")}")
    except Exception as e :
        pprint(f"Error Occured !!{e}")
    return 


tokens_saved : list[str] = []
opened_data = open_reads_json_file(os.getenv('DATA_SET_FILE'))

word_sets = start_word_set_generation(opened_data)
recur_func(tokens_saved,word_sets, 0, 250)