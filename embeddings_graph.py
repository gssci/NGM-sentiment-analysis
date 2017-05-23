import pickle


class EmbeddingsGraph:

    def __init__(self):
        self.graph = pickle.load(open("./data/graph/graph.p", "rb"))
        self.edges_ll = pickle.load(open("./data/graph/edges_ll.p", "rb"))
        self.edges_lu = pickle.load(open("./data/graph/edges_lu.p", "rb"))
        self.edges_uu = pickle.load(open("./data/graph/edges_uu.p", "rb"))
        self.indices_dict = pickle.load(open("./data/graph/indices_dict.p", "rb"))
        self.edges_weights = pickle.load(open("./data/graph/edges_weights.p", "rb"))

    def weight(self,u,v):
        if u < v:
            return self.edges_weights.get((u,v))
        else:
            return self.edges_weights.get((v,u))

    def get_file_name(self, edge):
        return self.indices_dict.get(edge)