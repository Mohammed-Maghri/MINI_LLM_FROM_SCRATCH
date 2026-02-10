
import os
from dotenv import load_dotenv
from tokenizer.opened_files import (open_reads_json_file, read_text_file)

load_dotenv()
def encoder_ (word : str) -> list[int] :
    instructions = read_text_file(os.getenv('INSTRUCTIONS_FILE'))
    splited_word : list[str] = []
    encoded_word : list[int] = []

    for i in range(word.__len__()) :
        splited_word.append(word[i])
    index : int = 0

    for instruction in instructions :
        encoded_word = instruction.split()
        # print(f"encoded word : {encoded_word}")
        
        # Skip if instruction doesn't have exactly 2 tokens
        if len(encoded_word) != 2:
            continue
            
        index = 0
        while index < splited_word.__len__() - 1 :
            if splited_word[index] == encoded_word[0] and splited_word[index + 1] == encoded_word[1] :
                splited_word[index] = encoded_word[0] + encoded_word[1]
                del splited_word[index + 1]
            index += 1
    return splited_word

def decode_tokens (encoded_tokens : list[str]) -> list[int] :
    decoded_tokens : list[int] = []
    for token in encoded_tokens :
        decoded_tokens.append(open_reads_json_file(os.getenv('FILE_GENERATION')).get("tokens_generated", {}).get(token, None))
    return decoded_tokens