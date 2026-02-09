def normalizer_ (text : str) -> list[str] :
    return text.split()

def generate_tokens_list (tokens : list[str]) -> dict :
    generated_tokens : dict = {"tokens_generated" : {}}
    try : 
        for i in range(255) :
            generated_tokens.get("tokens_generated", {})[bytes([i]).decode("latin-1")] = i
        for i in range(len(tokens)) :
            generated_tokens.get("tokens_generated", {})[tokens[i]] = i + 256
    except Exception as e :
        print(f"Error Occured !!{e}")
    return generated_tokens
