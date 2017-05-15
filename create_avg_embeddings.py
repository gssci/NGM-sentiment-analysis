"""
This script uses the Google News Word2Vec corpus to calculate
the average embeddings of each review in our dataset. 
It stores them into sparse matrices that I can then reuse in another script
"""
import logging
import gensim
import numpy as np
from scipy.spatial.distance import cosine
import scipy as sp
import glob
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)

#the elements of both matrices below constitute the nodes of our graph
#matrix of labeled embeddings
L = sp.sparse.lil_matrix((25000,300))

#matrix of unlabeled embeddings
U = sp.sparse.lil_matrix((50000,300))

def word2vec(w):
    out = np.zeros(300)
    try:
        out = model.word_vec(w)
    finally:
        return out

i = 0
for review in glob.glob('./data/train/pos/*.txt'):
    #read file of training data and correctly returns text
    #that includes html tags
    with open(review, 'r', encoding='utf8') as myfile:
        data = BeautifulSoup(myfile).get_text()
    #NLTK function to extract words from text
    words = word_tokenize(data)
    L[i] = np.mean([word2vec(w) for w in words], axis=0)
    print(str(i))
    i = i+1

for review in glob.glob('./data/train/neg/*.txt'):
    with open(review, 'r', encoding='utf8') as myfile:
        data = BeautifulSoup(myfile).get_text()
    words = word_tokenize(data)
    L[i] = np.mean([word2vec(w) for w in words], axis=0)
    print(str(i))
    i = i+1

#exports matrix to be used later in another script
sp.sparse.save_npz('labeled.npz', L.tocsr())

j = 0 
for review in glob.glob('./data/train/unsup/*.txt'):
    with open(review, 'r', encoding='utf8') as myfile:
        data = BeautifulSoup(myfile).get_text()
    words = word_tokenize(data)
    U[j] = np.mean([word2vec(w) for w in words], axis=0)
    print(str(j))
    j = j+1

sp.sparse.save_npz('unlabeled.npz', U.tocsr())