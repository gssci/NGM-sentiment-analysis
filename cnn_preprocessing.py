import pickle

class CNNPreprocessing(object):

    def __init__(self):
        self.indices_dict = pickle.load(open("./data/graph/indices_dict.p", "rb"))
        self.data_to_graph_index = {v: k for k, v in self.indices_dict.items()}
        self.alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n"
        self.length = 1014 # fixed size of character encoding


