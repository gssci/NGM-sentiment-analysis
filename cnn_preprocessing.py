import pickle
from bs4 import BeautifulSoup
import numpy as np
import scipy as sp
import scipy.sparse

indices_dict = pickle.load(open("./data/graph/indices_dict.p", "rb"))
data_to_graph_index = {v: k for k, v in indices_dict.items()}
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n"
length = 1014 # fixed size of character encoding


def extract_end(char_seq):
    if len(char_seq) > 1014:
        char_seq = char_seq[-1014:]
    return char_seq


def pad_sentence(char_seq, padding_char=" "):
    char_seq_length = 1014
    num_padding = char_seq_length - len(char_seq)
    new_char_seq = char_seq + str(padding_char) * num_padding
    return new_char_seq


def string_to_int8_conversion(char_seq):
    x = np.array([alphabet.find(char) for char in char_seq], dtype=np.int8)
    return x


def encode_review(index):
    """
    Transforms a review in a matrix of size 70x1014 of one-hot encodings
    70 are the features (characters in our accepted alphabet) 
    1014 are the number of characters in our string
    Any character exceeding length 1014 is ignore, and any characters that are 
    not in the alphabet including blank characters are encoded as all-zero vectors
    :param index: number of the review we need to fetch
    :return: encoded matrix to be used as input in the CNN 
    """
    out = sp.sparse.lil_matrix((len(alphabet),length),dtype=np.int8)
    path = indices_dict.get(index)
    with open(path, 'r', encoding='utf8') as file:
        input_string = BeautifulSoup(file, "html.parser").get_text()

    input_string = input_string.lower()
    input_string = extract_end(input_string)
    input_string = pad_sentence(input_string)

    x = string_to_int8_conversion(input_string)

    for i in range(len(x)):
        if x[i] != -1:
            out[x[i],i] = 1

    return out