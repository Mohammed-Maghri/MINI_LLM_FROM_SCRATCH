from pprint import pprint
from functions.opened_files import open_json_file
from functions.byte_pair_encoding import ( 
    start_word_set_generation,
    recur_func )

tokens_saved : list[str] = []
opened_data = open_json_file('./data_set/data.json')
word_sets = start_word_set_generation(opened_data)
recur_func(tokens_saved,word_sets, 0, 20)
pprint(tokens_saved)