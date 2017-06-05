"""
This script uses the Google News Word2Vec corpus to calculate
the average embeddings of each review in our dataset. 
It stores them into sparse matrices that I can then reuse in another script
"""
import logging
import gensim
import numpy as np
import scipy as sp
import scipy.sparse
import glob
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model = gensim.models.KeyedVectors.load_word2vec_format('./word2vec_model/GoogleNews-vectors-negative300.bin', binary=True)

# the elements of both matrices below constitute the nodes of our graph
# matrix of labeled embeddings
L = sp.sparse.lil_matrix((25000,300))

# matrix of unlabeled embeddings
U = sp.sparse.lil_matrix((50000,300))

def word2vec(w):
    """
    with this quick trick I can calculate the embeddings without normalizing the text (removing puctuaction, stop words etc...)
    If I pass a word that is not in the word2vec_model, like a stopword or some weird symbol, it just returns a zero vector that
    does not cotribute to the avg embedding
    """
    out = np.zeros(300)
    try:
        out = model.word_vec(w)
    finally:
        return out

i = 0
for review in glob.glob('./data/train/pos/*.txt'):
    # read file of training raw_data and correctly returns text
    # that includes html tags
    with open(review, 'r', encoding='utf8') as myfile:
        data = BeautifulSoup(myfile, "html5lib").get_text()
    # NLTK function to extract words from text
    # very important, much better than splitting the string
    words = word_tokenize(data)
    
    # embedding for review is calculated as average of the embeddings of all words
    # this is not ideal but is shown to work reasonably well in literature
    # if you need something a bit more sophisticated, look into Doc2Vec algorithms
    L[i] = np.mean([word2vec(w) for w in words], axis=0)
    print(str(i))
    i = i+1

for review in glob.glob('./data/train/neg/*.txt'):
    with open(review, 'r', encoding='utf8') as myfile:
        data = BeautifulSoup(myfile, "html5lib").get_text()
    words = word_tokenize(data)
    L[i] = np.mean([word2vec(w) for w in words], axis=0)
    print(str(i))
    i = i+1

# exports matrix to be used later in another script
sp.sparse.save_npz('./data/graph/labeled.npz', L.tocsr())

j = 0 
for review in glob.glob('./data/train/unsup/*.txt'):
    with open(review, 'r', encoding='utf8') as myfile:
        data = BeautifulSoup(myfile, "html5lib").get_text()
    words = word_tokenize(data)
    U[j] = np.mean([word2vec(w) for w in words], axis=0)
    print(str(j))
    j = j+1

sp.sparse.save_npz('./data/graph/unlabeled.npz', U.tocsr())