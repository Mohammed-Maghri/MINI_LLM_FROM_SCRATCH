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


def replace_voted_token(word_sets : dict, voted_token : str) -> dict:
    new_data_set : list[str] = []
    for word_set in word_sets.get("word_sets", []):
        word_correction : list[str] = word_set

        i = 0
        while i < len(word_correction) - 1:
            if ((word_correction[i] + word_correction[i + 1]) == voted_token):
                word_correction[i] = voted_token
                word_correction.pop(i + 1)
            i+= 1
        new_data_set.append(word_correction)

    word_sets["word_sets"] = new_data_set
    return word_sets


def sliding_window_comparison (word_sets : dict) -> list[str]:
    all_tokens = []
    tokens_vote = []
    for word_set in word_sets.get("word_sets", []):
        for i in range(len(word_set) - 1):
            if (i == len(word_set) - 1):
                pairs = (word_set[i], '_')
            else:                
                pairs = (word_set[i], word_set[i + 1])
                pprint(f"{pairs} --- {word_set}")
            tokens_vote.append(pairs)

    tokens_count = Counter(tokens_vote)
    tokens_vote.clear()

    all_list : list[int] = list(tokens_count.values())
    max_value : int = sorted(all_list)[len(all_list) - 1]

    for (key, value) in tokens_count.items():
        if value == max_value and key[1] != '_':
            all_tokens.append(key[0] + key[1])
    return all_tokens


def recur_func (open_data : dict, counter : int) -> None :
    if (open_data.get("tokens_size") == counter):
        return 
    word_sets = start_word_set_generation(opened_data)
    tokens = sliding_window_comparison(word_sets)
    dict_result = replace_voted_token(word_sets, min(tokens))
    pprint(min(tokens))
    pprint(dict_result)
    recur_func(dict_result, counter + 1)


opened_data = open_json_file('./data_set/data.json')
recur_func(opened_data, 0)
