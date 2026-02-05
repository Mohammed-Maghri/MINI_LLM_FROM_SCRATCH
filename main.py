import json
from pprint import pprint
from collections import Counter

def open_json_file(file_path : str) -> dict:
    try : 
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return {}

def start_word_set_generation(data : dict) -> dict:
    word_sets = []
    for key , values in opened_data.items():
        if isinstance(values, list):
            for value in values:
                element : list[str] = list(value)
                word_sets.append(element)
    return {"word_sets": word_sets,"tokens_size": opened_data.get("tokens_size", 0)}

def replace_voted_token(word_sets : dict, voted_token : str) -> None:
    for word_set in word_sets.get("word_sets", []):
        word_correction : list[str] = word_set
        for i in range(len(word_correction) - 1):
            if ((word_set[i] + word_set[i + 1]) == voted_token):
                word_correction.pop(i)
                word_correction.pop(i + 1)
                i = 0
                pprint(f" !!!!= {word_correction} {i} {i + 1}")

            # word_set[i] = voted_token
        word_correction.clear
        print(f" ==? {word_set}")
    return None

def sliding_window_comparison (word_sets : dict, voted_token : str) -> list[str]:
    all_tokens = []
    tokens_vote = []

    replace_voted_token(word_sets, voted_token)
    for word_set in word_sets.get("word_sets", []):
        for i in range(len(word_set) - 1):
            if (i == len(word_set) - 1):
                pairs = (word_set[i], '_')
            else:                
                pairs = (word_set[i], word_set[i + 1])
            tokens_vote.append(pairs)

    tokens_count = Counter(tokens_vote)
    tokens_vote.clear()

    all_list : list[int] = list(tokens_count.values())
    max_value : int = sorted(all_list)[len(all_list) - 1]

    pprint(tokens_count.keys())
    for (key, value) in tokens_count.items():
        if value == max_value and key[1] != '_':
            all_tokens.append(key[0] + key[1])
    return all_tokens


opened_data = open_json_file('./data_set/data.json')
word_sets = start_word_set_generation(opened_data)

tokens = sliding_window_comparison(word_sets, 'ab')



# print(tokens)



