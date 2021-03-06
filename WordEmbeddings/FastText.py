import numpy as np
import pickle as pkl
from util import BuildData, RFC, RFCs
from gensim.models import FastText


xtrain, xtest, ytrain, ytest = BuildData('tweets_clean.pkl')


ftmodel = FastText(vector_size=200, min_count=5)
ftmodel.build_vocab(xtrain)
ftmodel.train(xtrain, total_examples=ftmodel.corpus_count, epochs=10)


xtrain_ft = []
xtest_ft = []


for i in xtrain:
    words = i.split( )
    word_vecft = np.zeros((10000, 200))
    for n in range(len(words)):
        if words[n] in ftmodel.wv:
            word_vecft[n] = ftmodel.wv[words[n]]
    xtrain_ft.append(np.reshape(word_vecft,(2000000)))

for i in xtest:
    word_vecft = np.zeros((10000, 200))
    words = i.split( )
    for n in range(len(words)):
        if words[n] in ftmodel.wv:
            word_vecft[n] = ftmodel.wv[words[n]]
    xtest_ft.append(np.reshape(word_vecft,(2000000)))
    
    
acc = RFCs(xtrain_ft, ytrain, xtest_ft, ytest)
