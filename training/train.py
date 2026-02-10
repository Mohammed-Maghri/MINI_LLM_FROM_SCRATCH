from tokenizer.encoder import (encoder_ , decode_tokens)
import numpy as np
from tokenizer.opened_files import open_reads_json_file
import os
from dotenv import load_dotenv

load_dotenv()

sentences : list[str] = [
    "the cat sits on the mat",
    # "the dog sits on the rug",
    # "the cat eats fish",
    # "the dog eats meat",
    # "the cat drinks water",
    # "the dog drinks water",
    # "cats and dogs are animals",
    # "a cat is an animal",
    # "a dog is an animal",
    # "animals need food and water"
]


def build_dataset(tokens, context):
    X, Y = [], []

    for i in range(len(tokens) - context):
        print(' XX --- >< ' , tokens[i:i+context])
        print(' YY --- >< ' , tokens[i+1:i+context+1])
        X.append(decode_tokens(tokens[i:i+context]))
        Y.append(decode_tokens(tokens[i+1:i+context+1]))

    return np.array(X), np.array(Y)



for sentence in sentences:
    encoded_tokens = encoder_(sentence)
    context_size = 4
    X, Y = build_dataset(encoded_tokens, context_size)
    print(f"Input (X): {X}")
    print(f"Output (Y): {Y}")


Enb = np.random.randn(len(open_reads_json_file(os.getenv('FILE_GENERATION')).get("tokens_generated", {})), 64) / np.sqrt(64)

print(Enb)