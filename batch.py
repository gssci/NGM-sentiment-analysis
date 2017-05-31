import numpy as np
import networkx as nx
from embeddings_graph import EmbeddingsGraph
import random

graph = EmbeddingsGraph().G
X = np.load("./data/train/encoded_samples.npy")


def label(i):
    if 0 <= i < 12500:
        return np.array([1, 0])
    else:
        return np.array([0, 1])


def next_batch(h_edges, start, finish):
    """
    Helper function for the iterator, note that the neural graph machines,
    due to its unique loss function, requires carefully crafted inputs

    Refer to the Neural Graph Machines paper, section 3 and 3.3 for more details
    """
    edges_ll = list()
    edges_lu = list()
    edges_uu = list()
    w_ll = list()
    w_lu = list()
    w_uu = list()
    edg = h_edges[start:finish]
    edg = np.asarray(edg)

    for i, j in edg[:]:
        if (0 <= i < 25000) and (0 <= j < 25000):
            edges_ll.append((i, j))
            w_ll.append(graph.get_edge_data(i,j)['weight'])
        elif (0 <= i < 25000) and (25000 <= j < 75000):
            edges_lu.append((i, j))
            w_lu.append(graph.get_edge_data(i,j)['weight'])
        else:
            edges_uu.append((i, j))
            w_uu.append(graph.get_edge_data(i,j)['weight'])

    sub_ell = nx.Graph(data=edges_ll)
    sub_elu = nx.Graph(data=edges_lu)

    #convert to arrays for easier slicing
    edges_ll = np.asarray(edges_ll)
    edges_lu = np.asarray(edges_lu)
    edges_uu = np.asarray(edges_uu)

    u_ll = edges_ll[:, 0]
    #number of incident edges for nodes u
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
        Generates a batch iterator for the dataset.
    """
    # data = np.array(data)

    data_size = len(graph.edges())
    num_batches_per_epoch = int(data_size / batch_size) + 1
    for epoch in range(num_epochs):
        print("In epoch >> " + str(epoch + 1))
        print("num batches per epoch is: " + str(num_batches_per_epoch))

        # Shuffle the data at each epoch
        # Partition the graph into neighbourhood regions
        # and shuffle edges
        partitions = list()
        visited = set()
        for node in graph.nodes():
            if node not in visited:
                adjacent_edges = graph.edges(node)
                partitions.append(adjacent_edges)
                for u, v in adjacent_edges:
                    visited.add(u)
                    visited.add(v)

        random.shuffle(partitions)

        for partition in partitions:
            random.shuffle(partition)

        helper_edges = [val for sublist in partitions for val in sublist]

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield next_batch(helper_edges,start_index,end_index)