import pprint
from collections import Counter

def start_word_set_generation(data : dict) -> dict:
    word_sets = []
    for key , values in data.items():
        if isinstance(values, list):
            for value in values:
                element : list[str] = list(value)
                word_sets.append(element)
    return {"word_sets": word_sets,"tokens_size": data.get("tokens_size", 0)}


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
            pairs = (word_set[i], word_set[i + 1])
            tokens_vote.append(pairs)

    tokens_count = Counter(tokens_vote)
    tokens_vote.clear()

    if not tokens_count:
        return []

    all_list : list[int] = list(tokens_count.values())
    max_value : int = max(all_list)

    for (key, value) in tokens_count.items():
        if value == max_value:
            all_tokens.append(key[0] + key[1])
    return all_tokens


def recur_func (tokens_saved : list[str], word_set : dict, counter : int, max_tokens : int) -> None :
    try :
        if (counter == max_tokens) :
            return
        tokens = sliding_window_comparison(word_set)
        if not tokens:
            return

        selected_token = tokens[0]
        dict_result = replace_voted_token(word_set, selected_token)
        tokens_saved.append(selected_token)
        recur_func(tokens_saved , dict_result, counter + 1, max_tokens)
    except Exception as e:
        pprint(f" Error Occured ! {e}") 