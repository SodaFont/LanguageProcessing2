import numpy as np
import pickle as pkl
from util import BuildData, RFC, RFCs
from gensim.models.word2vec import Word2Vec


xtrain, xtest, ytrain, ytest = BuildData('tweets_clean.pkl')


w2vmodel = Word2Vec(vector_size=200, min_count=5)
w2vmodel.build_vocab(xtrain)
w2vmodel.train(xtrain, total_examples=w2vmodel.corpus_count, epochs=10)


xtrain_wv = []
xtest_wv = []


for i in xtrain:
    words = i.split( )
    word_vecwv = np.zeros((10000, 200))
    for n in range(len(words)):
        if words[n] in w2vmodel.wv:
            word_vecwv[n] = w2vmodel.wv[words[n]]
    xtrain_wv.append(np.reshape(word_vecwv,(2000000)))

for i in xtest:
    word_vecwv = np.zeros((10000, 200))
    words = i.split( )
    for n in range(len(words)):
        if words[n] in w2vmodel.wv:
            word_vecwv[n] = w2vmodel.wv[words[n]]
    xtest_wv.append(np.reshape(word_vecwv,(2000000)))
    
    
acc = RFCs(xtrain_wv, ytrain, xtest_wv, ytest)
