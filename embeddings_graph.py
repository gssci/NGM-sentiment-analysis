import pickle
import networkx as nx
import random


class EmbeddingsGraph:

    def __init__(self):
        self.graph = nx.Graph()
        #self.graph = pickle.load(open("./data/graph/graph.p", "rb"))
        edges_ll = pickle.load(open("./data/graph/edges_ll.p", "rb"))
        edges_lu = pickle.load(open("./data/graph/edges_lu.p", "rb"))
        edges_uu = pickle.load(open("./data/graph/edges_uu.p", "rb"))
        self.edges = edges_ll + edges_lu + edges_uu
        self.indices_dict = pickle.load(open("./data/graph/indices_dict.p", "rb"))
        self.edges_weights = pickle.load(open("./data/graph/edges_weights.p", "rb"))

        for (u,v) in self.edges:
            self.graph.add_edge(u, v, weight=self.edges_weights.get((u, v)))

    def weight(self,u,v):
        if u < v:
            return self.edges_weights.get((u,v))
        else:
            return self.edges_weights.get((v,u))

    def get_file_name(self, edge):
        return self.indices_dict.get(edge)
