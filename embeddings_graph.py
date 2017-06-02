import pickle
import networkx as nx
import random


class EmbeddingsGraph:

    def __init__(self):
        self.G = nx.Graph()
        self.ELL = nx.Graph()
        self.ELU = nx.Graph()
        self.EUU = nx.Graph()
        self.graph = pickle.load(open("./data/graph/graph.p", "rb"))
        self.edges_ll = pickle.load(open("./data/graph/edges_ll.p", "rb"))
        self.edges_lu = pickle.load(open("./data/graph/edges_lu.p", "rb"))
        self.edges_uu = pickle.load(open("./data/graph/edges_uu.p", "rb"))
        self.edges = self.edges_ll + self.edges_lu + random.sample(self.edges_uu,30000)
        self.indices_dict = pickle.load(open("./data/graph/indices_dict.p", "rb"))
        self.edges_weights = pickle.load(open("./data/graph/edges_weights.p", "rb"))

        for (u,v) in self.edges:
            self.G.add_edge(u,v,weight=self.edges_weights.get((u,v)))

        # for (u,v) in self.edges_ll:
        #     self.ELL.add_edge(u,v,weight=self.edges_weights.get((u,v)))
        #
        # for (u,v) in self.edges_lu:
        #     self.ELU.add_edge(u,v,weight=self.edges_weights.get((u,v)))
        #
        # for (u,v) in self.edges_uu:
        #     self.EUU.add_edge(u,v,weight=self.edges_weights.get((u,v)))

    def weight(self,u,v):
        if u < v:
            return self.edges_weights.get((u,v))
        else:
            return self.edges_weights.get((v,u))

    def get_file_name(self, edge):
        return self.indices_dict.get(edge)