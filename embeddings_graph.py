"""
Script for generating the data structures for the embeddings graph to be used
in the Conv-NN optimization function
"""
import pickle

class embeddings_graph:

    def __init__(self):
        self.edges_LL = pickle.load(open("./data/graph/edges_LL.p", "rb"))
        self.edges_LU = pickle.load(open("./data/graph/edges_LU.p", "rb"))
        self.edges_UU = pickle.load(open("./data/graph/edges_UU.p", "rb"))
        self.indices_dict = pickle.load(open("./data/graph/indices_dict.p", "rb"))
        self.edges_weights = pickle.load(open("./data/graph/edges_weights.p", "rb"))

    def edge_weight(self,u,v):
        if u < v:
            return self.edges_weights.get(u,v)
        else:
            return self.edges_weights.get(v,u)

