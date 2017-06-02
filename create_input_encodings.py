"""
Simple script for generating the character level encoding of my input data, and then save it to
disk for later usage
"""
import pickle
import numpy as np
from bs4 import BeautifulSoup
import glob

indices_dict = pickle.load(open("./data/graph/indices_dict.p", "rb"))
data_to_graph_index = {v: k for k, v in indices_dict.items()}
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n"
length = 1014 # fixed size of character encoding


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
    char_seq = char_seq.lower()
    char_seq = extract_end(char_seq)
    char_seq = pad_sentence(char_seq)
    return np.array([alphabet.find(char) for char in char_seq], dtype=np.int8)


def encode_traing_samples():
    n_samples = len(indices_dict.keys())
    X = np.zeros((n_samples, 1014), dtype=np.int8)

    for i in range(n_samples):
        path = indices_dict.get(i)
        with open(path, 'r', encoding='utf8') as file:
            input_string = BeautifulSoup(file, "html.parser").get_text()

        X[i] = string_to_int8_conversion(input_string)
        print(str(i))

    np.save("./data/train/encoded_samples.npy", X)


def encode_test_samples():
    X_test = np.zeros((25000, 1014), dtype=np.int8)
    Y_test = np.zeros((25000, 2), dtype=np.int8)
    i = 0

    for review in glob.glob('./data/test/pos/*.txt'):
        with open(review, 'r', encoding='utf8') as myfile:
            data = BeautifulSoup(myfile).get_text()

        X_test[i] = string_to_int8_conversion(data)
        Y_test[i, 1] = 1
        print(str(i))
        i = i+1

    for review in glob.glob('./data/test/neg/*.txt'):
        with open(review, 'r', encoding='utf8') as myfile:
            data = BeautifulSoup(myfile).get_text()

        X_test[i] = string_to_int8_conversion(data)
        Y_test[i, 0] = 1
        print(str(i))
        i = i+1

    np.save("./data/test/test_encoded_samples.npy", X_test)
    np.save("./data/test/test_labels.npy", Y_test)