#!/usr/bin/env python
# coding: utf-8

# In[1]:


import io, sys, math, re
from collections import defaultdict
import numpy as np


# In[2]:


# dataloader

def load_data(filename):
    fin = io.open(filename, 'r', encoding='utf-8')
    data = []
    vocab = defaultdict(lambda:0)
    for line in fin:
        sentence = line.split()
        data.append(sentence)
        for word in sentence:
            vocab[word] += 1
    return data, vocab


# In[3]:


def remove_rare_words(data, vocab, mincount = 10):
    ## FILL CODE
    # replace words in data that are not in the vocab 
    # or have a count that is below mincount
    data_with_unk = data[:]
    
    for i in range(len(data)):
        for word in range(len(data[i])):
            if vocab[data[i][word]] < mincount:
                data_with_unk[i][word] = "unk"
                
    return data_with_unk


# In[4]:


# LOAD DATA

print("load training set")
train_data, vocab = load_data("train.txt")

## FILL CODE
# Same as bigram.py
remove_rare_words(train_data, vocab)

print("load validation set")
valid_data, _ = load_data("valid.txt")
remove_rare_words(valid_data, vocab)
## FILL CODE
# Same as bigram.py


# In[5]:


def build_ngram(data, n):
    total_number_words = 0
    counts = defaultdict(lambda: defaultdict(lambda: 0.0))

    for sentence in data:
        sentence = tuple(sentence)
        ## FILL CODE
        # dict can be indexed by tuples
        # store in the same dict all the ngrams
        # by using the context as a key and the word as a value
        for i in range(len(sentence)):
            temp = sentence[i:n+i]
            for j in range(len(temp)):
                counts[tuple(temp[:j])][temp[j]] += 1

    prob  = defaultdict(lambda: defaultdict(lambda: 0.0))
    ## FILL CODE
    # Build the probabilities from the counts
    # Be careful with how you normalize!
    for p in counts:
        s = sum(counts[p].values())
        for w in counts[p]:
            prob[p][w] = 1.0 * counts[p][w] / s

    return prob


# In[23]:


# RUN TO BUILD NGRAM MODEL

n = 4
print("build ngram model with n = ", n)
model = build_ngram(train_data, n)


# # Perplexity

# In[24]:


def get_prob(model, context, w):
    ## FILL CODE
    # code a recursive function over 
    # smaller and smaller context
    # to compute the backoff model
    # Bonus: You can also code an interpolation model this way
    #print(tuple(context), w)
    if model[tuple(context)][w] != 0:
        return model[tuple(context)][w]
    else:
        return 0.4 * get_prob(model, context[1:], w)

def perplexity(model, data, n):
    ## FILL CODE
    # Same as bigram.py
    T = 0
    log_sum = 0
    
    for sentence in data:
        prev_word = sentence[:n-1]
        for word in sentence[n-1:]:
            log_sum += np.log(get_prob(model, prev_word, word))
            if prev_word:
                prev_word.pop(0)
                prev_word.append(word)
        T+= len(sentence)
    perp = -(log_sum/T)
    return perp


# In[25]:


# COMPUTE PERPLEXITY ON VALIDATION SET

print("The perplexity is", perplexity(model, valid_data, n))


# In[26]:


def get_proba_distrib(model, context):
    ## FILL CODE
    # code a recursive function over context
    # to find the longest available ngram  
    if sum(model[tuple(context)].values()) > 0:
        return  model[tuple(context)]
    else:
        return get_proba_distrib(model, context[1:])  

def generate(model):
    ## FILL CODE
    # generate a sentence. A sentence starts with a <s> and ends with a </s>
    # Possiblly a use function is:
    #   np.random.choice(x, 1, p = y)
    # where x is a list of things to sample from
    # and y is a list of probability (of the same length as x)
    sentence = ["<s>"]
    while True :
        world_dict = get_proba_distrib(model, sentence)
        p = np.random.choice(list(world_dict.keys()), 1, p = list(world_dict.values()))[0]
        sentence.append(p)
        if p == "</s>": break 
    
    return sentence


# In[27]:


# GENERATE A SENTENCE FROM THE MODEL

print("Generated sentence: ",generate(model))


# In[ ]:




