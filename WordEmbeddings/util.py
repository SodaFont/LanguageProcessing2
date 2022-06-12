import numpy as np
from sklearn.ensemble import RandomForestClassifier

 
    
def BuildData(file):
    with open(file, 'rb') as f:
    data = pkl.load(f)
    
    xtrain = []
    xtest = []
    ytrain = []
    ytest = []

    for i in data[:336]:
        xtrain.append(i[1])
        ytrain.append(i[2])

    for i in data[336:]:
        xtest.append(i[1])
        ytest.append(i[2])
        
    return xtrain, ytrain, xtest, ytest
    
    
    
def RFC(n, xtrain, ytrain, xtest, ytest):
    rfc = RandomForestClassifier(n_estimators=n).fit(xtrain, ytrain)
    score_rf = rfc.score(xtest, ytest)
    print('The classification accuracy with ' + str(n) +' trees is', score_rf)
    return score_rf
    
def RFCs(xtrain, ytrain, xtest, ytest):
    acc = []
    for n in [30, 50, 80, 100, 200, 500]:
        acc.append(RFC(n, xtrain, ytrain, xtest, ytest))
    return acc
