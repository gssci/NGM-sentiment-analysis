import pickle
from bs4 import BeautifulSoup
import numpy as np
import scipy as sp
import scipy.sparse
from embeddings_graph import EmbeddingsGraph
import networkx as nx

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
    char_seq = char_seq.lower()
    char_seq = extract_end(char_seq)
    char_seq = pad_sentence(char_seq)
    return np.array([alphabet.find(char) for char in char_seq], dtype=np.int8)

X = np.load("./data/train/encoded_samples.npy")
edges = g.edges

def label(i):
    if 0 <= i < 12500:
        return np.array([1, 0])
    else:
        return np.array([0, 1])


def next_batch(bedges,start,finish):
    edges_ll = []
    edges_lu = []
    edges_uu = []
    w_ll = []
    w_lu = []
    w_uu = []
    edg = bedges[start:finish]
    edg = np.asarray(edg)

    for i, j in edg[:]:
        if (0 <= i < 25000) and (0 <= j < 25000):
            edges_ll.append((i, j))
            w_ll.append(g.weight(i,j))
        elif (0 <= i < 25000) and (25000 <= j < 75000):
            edges_lu.append((i, j))
            w_lu.append(g.weight(i,j))
        else:
            edges_uu.append((i, j))
            w_uu.append(g.weight(i,j))

    sub_ell = nx.Graph(data=edges_ll)
    sub_elu = nx.Graph(data=edges_lu)
    edges_ll = np.asarray(edges_ll)
    edges_lu = np.asarray(edges_lu)
    edges_uu = np.asarray(edges_uu)

    u_ll = edges_ll[:, 0]
    c_ull = [1 / len(sub_ell.edges(u)) for u in u_ll]
    v_ll = edges_ll[:, 1]
    c_vll = [1 / len(sub_ell.edges(v)) for v in v_ll]
    u1 = X[u_ll]
    lu1 = np.vstack([label(u) for u in u_ll])
    v1 = X[v_ll]
    lv1 = np.vstack([label(v) for v in v_ll])

    u_lu = edges_lu[:, 0]
    c_ulu = [1 / len(sub_elu.edges(u)) for u in u_lu]
    u2 = X[u_lu]
    lu2 = np.vstack([label(u) for u in u_lu])
    v2 = X[edges_lu[:, 1]]

    u3 = X[edges_uu[:, 0]]
    v3 = X[edges_uu[:, 1]]
    return u1, v1, lu1, lv1, u2, v2, lu2, u3, v3, w_ll, w_lu, w_uu, c_ull, c_vll, c_ulu

def batch_iter(batch_size, num_epochs):
    """
        Generates a batch iterator for a dataset.
        """
    # data = np.array(data)

    data_size = len(edges)
    num_batches_per_epoch = int(data_size / batch_size) + 1
    for epoch in range(num_epochs):
        print("In epoch >> " + str(epoch + 1))
        print("num batches per epoch is: " + str(num_batches_per_epoch))
        # Shuffle the data at each epoch
        bedges = np.random.permutation(edges)
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield next_batch(bedges,start_index,end_index)

# if __name__ == '__main__':
#     n_samples = len(indices_dict.keys())
#     X = np.zeros((n_samples,1014),dtype=np.int8)
#
#     for i in range(n_samples):
#         path = indices_dict.get(i)
#         with open(path, 'r', encoding='utf8') as file:
#             input_string = BeautifulSoup(file, "html.parser").get_text()
#
#         X[i] = string_to_int8_conversion(input_string)
#         print(str(i))

