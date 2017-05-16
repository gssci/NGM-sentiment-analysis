"""
This script generates the dictionaries that I need for representing 
the graph connecting the review embeddings, and to access them later in constant time.
A node is only connected only to its closest neighbors.

The job is split into three processes for a reasonable speedup

For convenience I also save the indices corresponding to the files in the order 
they appear in the glob function, so that we know what number corresponds to what review.
"""
import numpy as np
import scipy as sp
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Process, Manager, Value, Lock
import time
import pickle
import glob

L = sp.sparse.load_npz('./data/graph/labeled.npz')
U = sp.sparse.load_npz('./data/graph/unlabeled.npz')
M = sp.sparse.vstack([L,U])
last_index_l = 25000
last_index_u = 75000

#we only keep the closest neighbors
max_neighs = 10
size = M.shape[0]

def compute_files_indices():
    """
    maps into a dictionary the index corresponding to the file
    used for calculating the embeddings
    """
    indices_dict = {}

    i = 0
    for review in glob.glob('./data/train/pos/*.txt'):
        indices_dict.update({i:review})
        i = i+1

    for review in glob.glob('./data/train/neg/*.txt'):
        indices_dict.update({i:review})
        i = i+1

    for review in glob.glob('./data/train/unsup/*.txt'):
        indices_dict.update({i:review})
        i = i+1

    pickle.dump(indices_dict, open( "./data/graph/indices_dict.p", "wb" ))
    return

def compute_graph_for_embedding(graph,edges_weights,edges_ll,edges_lu,edges_uu,chunk,counter,lock):
    """
    Function for computing the subgraph for nodes.
    Note that the edges_* structures are meant to be used later in the objective function of the Conv-NN
    and are not of any particular interest for the sake of the graph creation
    :param graph: a dictionary mapping a node to a list of neighbors
    :param edges_weights: dict that maps an egde (u,v) to its weight, in our case cosine_similarity
    :param edges_ll: edges from labeled node to labeled node
    :param edges_lu: edges from labeled node to unlabeled node
    :param edges_uu: edges from unlabeled node to unlabeled node
    :param chunk: range of values repressenting a chuck of embeddings (nodes) for which we want to find the neighbors
    :param counter: shared memory value used to track progress
    :param lock: lock to ensure atomicity of counter update
    """
    for i in chunk:
        sim = cosine_similarity(M[i],M)
        # sklearn outputs a matrix, we only need the row vector
        sim = sim[0]
        #set the embedding similarity with itself (==1) to zero
        sim[i] = 0

        neighbors_indices = list(sim.argsort()[-max_neighs::][::-1])
        correct_indices = [j for j in neighbors_indices if i < j]

        graph.update({i:correct_indices})

        n = len(correct_indices)

        if n > 0:
            edges = list(zip([i] * n, correct_indices))
            edges_weights.update(dict(zip(edges,np.take(sim,correct_indices))))

            for j in correct_indices:
                if (0 <= i < last_index_l) and (0 <= j < last_index_l):
                    edges_ll.append((i,j))
                elif (0 <= i < last_index_l) and (last_index_l <= j < last_index_u):
                    edges_lu.append((i,j))
                else:
                    edges_uu.append((i,j))
        with lock:
            counter.value += 1
            print(str(counter.value))
    return

if __name__ == '__main__':
    compute_files_indices()

    manager = Manager()
    graph = manager.dict()
    edges_weights = manager.dict()
    edges_ll = manager.list()
    edges_lu = manager.list()
    edges_uu = manager.list()

    counter = Value('i', 0)
    lock = Lock()

    processes = []

    #I split the job manually for my two-core laptop
    chunks = [range(0,20000),range(20000,45000),range(45000,75000)]

    for chunk in chunks:
        p = Process(target=compute_graph_for_embedding, args=(graph, edges_weights, edges_ll, edges_lu, edges_uu, chunk, counter, lock))
        processes += [p]

    _ = [p.start() for p in processes]
    _ = [p.join() for p in processes]

    #breathe
    time.sleep(2)

    #save to file the data structure that we worked so hard to compute
    pickle.dump(graph, open("./data/graph/graph.p", "wb"))
    pickle.dump(edges_weights, open("./data/graph/edges_weights.p", "wb"))
    pickle.dump(edges_ll, open("./data/graph/edges_ll.p", "wb"))
    pickle.dump(edges_lu, open("./data/graph/edges_lu.p", "wb"))
    pickle.dump(edges_uu, open("./data/graph/edges_uu.p", "wb"))