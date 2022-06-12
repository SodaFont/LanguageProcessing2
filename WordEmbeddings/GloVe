import numpy as np
import scipy
import sklearn
import pickle as pkl
from util import RFC, RFCs, BuildData
import torch
import torchtext

    
xtrain, xtest, ytrain, ytest = BuildData('tweets_clean.pkl')
    
    
embeddings_index = {}
embedding_dim = 100
f = open('glove.twitter.27B.100d.txt', encoding = "utf-8")
glove = torchtext.vocab.GloVe(name="twitter.27B", dim=200)  


xtrain_glove = []
a = 0
for i in xtrain:
    words = i.split( )
    word_vecglove = np.zeros((10000, 200))
    for n in range(len(words)):
        em = glove[words[n]].numpy()
        word_vecglove[n] = em
    xtrain_glove.append(np.reshape(word_vecglove,(2000000)))
    a += 1
    print(a)


xtest_glove = []
a = 0
for i in xtest:
    words = i.split( )
    word_vecglove = np.zeros((10000, 200))
    for n in range(len(words)):
        em = glove[words[n]].numpy()
        word_vecglove[n] = em
    xtest_glove.append(np.reshape(word_vecglove,(2000000)))
    a += 1
    print(a)
    
    
acc = RFCs(xtrain_glove, ytrain, xtest_glove, ytest)
