import pickle
from bs4 import BeautifulSoup
import numpy as np
import scipy as sp
import scipy.sparse
from embeddings_graph import EmbeddingsGraph

indices_dict = pickle.load(open("./data/graph/indices_dict.p", "rb"))
data_to_graph_index = {v: k for k, v in indices_dict.items()}
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n"
length = 1014 # fixed size of character encoding
g = EmbeddingsGraph()

def extract_end(char_seq):
    """all sequences longer than 1014 are ignored"""
    if len(char_seq) > 1014:
        char_seq = char_seq[-1014:]
    return char_seq


def pad_sentence(char_seq, padding_char=" "):
    """add blank spaces at the end of sequences shorter than 1014"""
    char_seq_length = 1014
    num_padding = char_seq_length - len(char_seq)
    new_char_seq = char_seq + str(padding_char) * num_padding
    return new_char_seq


def string_to_int8_conversion(char_seq):
    """
    returns an array of numbers representing the position of the
    feature or character in the sequence or -1 if it is not in our alphabet
    """
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

    path = indices_dict.get(index)
    with open(path, 'r', encoding='utf8') as file:
        input_string = BeautifulSoup(file, "html.parser").get_text()

    input_string = input_string.lower()
    input_string = extract_end(input_string)
    input_string = pad_sentence(input_string)

    x = string_to_int8_conversion(input_string)

    out = np.zeros(shape=[1, len(alphabet), len(x), 1],dtype=np.float32)
    #populate the encoding matrix, initially of all zeros
    for i in range(len(x)):
        if x[i] != -1:
            out[0][x[i]][i][0] = 1

    return out