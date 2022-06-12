import numpy as np
import pickle as pkl
from util import BuildData, RFC, RFCs
from transformers import GPT2LMHeadModel, GPT2Tokenizer

xtrain, xtest, ytrain, ytest = BuildData('author_sentences.pkl')

    
model = GPT2LMHeadModel.from_pretrained('gpt2')
word_embeddings = model.transformer.wte.weight
position_embeddings = model.transformer.wpe.weight
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


xtrain_gpt = []
a = 0
for i in xtrain:
    c = 0
    for j in i:
        words = j.split( )
        b = 0
        for n in range(len(words)):
            text_index = tokenizer.encode(words[n],add_prefix_space=True)
            vector = model.transformer.wte.weight[text_index[0],:]
            em = vector.detach().numpy()
            if b == 0:
                word_vecgpt = em
                b += 1
                # print(np.shape(word_vecgpt))
            elif b == 1:
                word_vecgpt = np.concatenate(([word_vecgpt],[em]), axis = 0)
                # print(np.shape(word_vecgpt))
                b += 1
            else:
                word_vecgpt = np.concatenate((word_vecgpt,[em]), axis = 0)
                # print(np.shape(word_vecgpt))
        # print(np.shape(word_vecgpt))
        if len(word_vecgpt) == 768:
            pass
        else:
            sen_vecgpt = np.mean(word_vecgpt, axis=0)
        # print(np.shape(sen_vecgpt))
        if c == 0:
            vecgpt = sen_vecgpt
            c += 1
        else:
            vecgpt = np.concatenate((vecgpt, sen_vecgpt), axis = None)
        # print(len(vecgpt)/768)    
    xtrain_gpt.append(vecgpt)
    a += 1
    print(a)
    
    
xtest_gpt = []
a = 0
for i in xtest:
    c = 0
    for j in i:
        words = j.split( )
        b = 0
        for n in range(len(words)):
            text_index = tokenizer.encode(words[n],add_prefix_space=True)
            vector = model.transformer.wte.weight[text_index[0],:]
            em = vector.detach().numpy()
            if b == 0:
                word_vecgpt = em
                b += 1
            elif b == 1:
                word_vecgpt = np.concatenate(([word_vecgpt],[em]), axis = 0)
                b += 1
            else:
                word_vecgpt = np.concatenate((word_vecgpt,[em]), axis = 0)
        if len(word_vecgpt) == 768:
            pass
        else:
            sen_vecgpt = np.mean(word_vecgpt, axis=0)
        if c == 0:
            vecgpt = sen_vecgpt
            c += 1
        else:
            vecgpt = np.concatenate((vecgpt, sen_vecgpt), axis = None)   
    xtest_gpt.append(vecgpt)
    a += 1
    print(a)
    
    
acc = RFCs(xtrain_gpt, ytrain, xtest_gpt, ytest)
