import numpy as np
import pickle as pkl
from transformers import BertTokenizer
import torch
from transformers import BertTokenizer, BertModel
from util import RFC, RFCs, BuildData


xtrain, xtest, ytrain, ytest = BuildData('author_sentence.pkl')
    
    
tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
sentence='I really enjoyed this movie a lot.'
#1.Tokenize the sequence:
tokens=tokenizer.tokenize(sentence)
print(tokens)
print(type(tokens))


model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True,)
model.eval()


def sen_bert(text): 
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]
    token_embeddings = torch.stack(hidden_states, dim=0)
    token_vecs_sum = []
    for token in token_embeddings:
        sum_vec = torch.sum(token[-4:], dim=0)
        token_vecs_sum.append(sum_vec)
    token_vecs = hidden_states[-2][0]
    sentence_embedding = torch.mean(token_vecs, dim=0)
    return sentence_embedding.numpy()
    
    
xtrain_bert = []
a = 0
for i in xtrain:
    b = 0
    for j in i:
        vec = sen_bert(j)
        if b == 0:
            word_vecbert = vec
            b += 1
        else:
            word_vecbert = np.concatenate((word_vecbert,vec),axis = None)
    xtrain_bert.append(word_vecbert)
    a += 1
    print(a)
    
    
xtest_bert = []
a = 0
for i in xtest:
    b = 0
    for j in i:
        vec = sen_bert(j)
        if b == 0:
            word_vecbert = vec
            b += 1
        else:
            word_vecbert = np.concatenate((word_vecbert,vec),axis = None)
    xtest_bert.append(word_vecbert)
    a += 1
    print(a)
    
    
acc = RFCs(xtrain_bert, ytrain, xtest_bert, ytest)
