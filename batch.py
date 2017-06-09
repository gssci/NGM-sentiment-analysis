import numpy as np
import networkx as nx
from embeddings_graph import EmbeddingsGraph

graph = EmbeddingsGraph().graph
X = np.load("./data/train/encoded_samples.npy")

pos = 12500 #last index of positive sample
l = 25000 #last index of labeled samples
u = 75000 #last index of all samples


def label(i):
    if 0 <= i < 12500:
        return np.array([0, 1]) #pos
    else:
        return np.array([1, 0]) #neg


def next_batch(h_edges, start, finish):
    """
    Helper function for the iterator, note that the neural graph machines,
    due to its unique loss function, requires carefully crafted inputs

    Refer to the Neural Graph Machines paper, section 3 and 3.3 for more details
    """
    edges_ll = list()
    edges_lu = list()
    edges_uu = list()
    weights_ll = list()
    weights_lu = list()
    weights_uu = list()
    batch_edges = h_edges[start:finish]
    batch_edges = np.asarray(batch_edges)

    for i, j in batch_edges[:]:
        if (0 <= i < l) and (0 <= j < l):
            edges_ll.append((i, j))
            weights_ll.append(graph.get_edge_data(i,j)['weight'])
        elif (0 <= i < l) and (l <= j < u):
            edges_lu.append((i, j))
            weights_lu.append(graph.get_edge_data(i,j)['weight'])
        else:
            edges_uu.append((i, j))
            weights_uu.append(graph.get_edge_data(i,j)['weight'])

    u_ll = [e[0] for e in edges_ll]

    # number of incident edges for nodes u
    c_ull = [1 / len(graph.edges(n)) for n in u_ll]
    v_ll = [e[1] for e in edges_ll]
    c_vll = [1 / len(graph.edges(n)) for n in v_ll]
    nodes_ll_u = X[u_ll]

    labels_ll_u = np.zeros((0,2))
    if len(nodes_ll_u) > 0:
        labels_ll_u = np.vstack([label(n) for n in u_ll])

    nodes_ll_v = X[v_ll]

    labels_ll_v = np.zeros((0,2))
    if len(nodes_ll_v) > 0:
        labels_ll_v = np.vstack([label(n) for n in v_ll])

    u_lu = [e[0] for e in edges_lu]
    c_ulu = [1 / len(graph.edges(n)) for n in u_lu]
    nodes_lu_u = X[u_lu]
    nodes_lu_v = X[[e[1] for e in edges_lu]]

    labels_lu = np.zeros((0,2))
    if len(nodes_lu_u) > 0:
        labels_lu = np.vstack([label(n) for n in u_lu])

    nodes_uu_u = X[[e[0] for e in edges_uu]]
    nodes_uu_v = X[[e[1] for e in edges_uu]]

    return nodes_ll_u, nodes_ll_v, labels_ll_u, labels_ll_v, \
           nodes_uu_u, nodes_uu_v, nodes_lu_u, nodes_lu_v, \
           labels_lu, weights_ll, weights_lu, weights_uu, \
           c_ull, c_vll, c_ulu


def batch_iter(batch_size):
    """
        Generates a batch iterator for the dataset.
    """

    data_size = len(graph.edges())

    edges = np.random.permutation(graph.edges())

    num_batches = int(data_size / batch_size)

    if data_size % batch_size > 0:
        num_batches = int(data_size / batch_size) + 1

    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield next_batch(edges,start_index,end_index)


def test_batch_inter(batch_size=250):
    """
    batch iterator for test data with labels
    """
    test_samples = np.load("./data/test/test_encoded_samples.npy")
    test_labels = np.load("./data/test/test_labels.npy")
    len_data = len(test_samples)

    shuffle_indices = np.random.permutation(range(len_data))

    num_batches = int(data_size / batch_size)

    if data_size % batch_size > 0:
        num_batches = int(data_size / batch_size) + 1

    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, len_data)

        batch_indices = shuffle_indices[start_index:end_index]
        input_x = test_samples[batch_indices]
        labels = test_labels[batch_indices]
        yield input_x, labels