#!/usr/bin/env python
# coding: utf-8

# In[32]:


import io, sys, math, re
from collections import defaultdict
import numpy as np


# In[33]:


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


# In[34]:


data, vocab = load_data("train2.txt")


# In[35]:


data[:3]


# In[36]:


vocab


# In[37]:


def remove_rare_words(data, vocab, mincount):
    ## FILL CODE
    # replace words in data that are not in the vocab 
    # or have a count that is below mincount
    data_with_unk = data[:]
    
    for i in range(len(data)):
        for word in range(len(data[i])):
            if vocab[data[i][word]] < mincount:
                data_with_unk[i][word] = "unk"
                
    return data_with_unk


# In[38]:


# LOAD DATA

train_data, vocab = load_data("train2.txt")
## FILL CODE 
# If you have a Out of Vocabulary error (OOV) 
# call the function "remove_rare_words" to replace 
# rare words with <unk> in the dataset
remove_rare_words(train_data, vocab, 100)

print("load validation set")
valid_data, _ = load_data("valid2.txt")
remove_rare_words(valid_data, vocab, 100)

# If you have a Out of Vocabulary error (OOV) 
# call the function "remove_rare_words" to replace 
# OOV with <unk> in the dataset


# In[39]:


# Function to build a bigram model

def build_bigram(data):
    unigram_counts = defaultdict(lambda:0)
    bigram_counts  = defaultdict(lambda: defaultdict(lambda: 0.0))
    total_number_words = 0

    ## FILL CODE
    # Store the unigram and bigram counts as well as the total 
    # number of words in the dataset
    
    #unigram
    for sentence in data:
        for idx, word in enumerate(sentence):
            unigram_counts[word] += 1
            total_number_words+=1
            if idx < len(sentence)-1:
                bigram_counts[word][sentence[idx+1]] += 1
        

    unigram_prob = defaultdict(lambda:0)
    bigram_prob = defaultdict(lambda: defaultdict(lambda: 0.0))

    ## FILL CODE
    # Build unigram and bigram probabilities from counts
    
    for sentence in data:
        for idx, word in enumerate(sentence):
            unigram_prob[word] = (1.*unigram_counts[word])/total_number_words
            if idx<len(sentence) -1:
                bigram_prob[word][sentence[idx+1]] = (1.*bigram_counts[word][sentence[idx+1]])/unigram_counts[word]

    return {'bigram': bigram_prob, 'unigram': unigram_prob}


# In[40]:


# RUN TO BUILD BIGRAM MODEL

print("build bigram model")
model = build_bigram(train_data)
model


# # Perplexity
# 
# In NLP, the perplexity metric is a way to capture the degree of 'uncertainty' a model has in predicting(assigning probabilities to)some text. The lower the perplexity, the higher the probability hence the better the model. Related to [Shannon's entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory))

# In[41]:


def get_prob(model, w1, w2):
    assert model["unigram"][w2] != 0, "Out of Vocabulary word!"
    ## FILL CODE
    # Should return the probability of the bigram (w1w2) if it exists
    # Else it return the probility of unigram (w2) multiply by 0.4
    
    prob = model["bigram"][w1][w2]
    
    if prob == 0:
        prob = 0.4 * model["unigram"][w2]
    
    return prob

def perplexity(model, data):
    ## FILL CODE
    # follow the formula in the slides
    # call the function get_prob to get P(w2 | w1)
    T = 0
    log_sum = 0
    
    for sentence in data:
        prev_word = sentence[0]
        for word in sentence[1:]:
            log_sum += np.log(get_prob(model, prev_word, word))
            prev_word = word
        T+= len(sentence)
    perp = -(log_sum/T)
    return perp


# In[42]:


# COMPUTE PERPLEXITY ON VALIDATION SET
print("The perplexity is", perplexity(model, valid_data))


# In[43]:


def generate(model):
    sentence = ["<s>"]
    ## FILL CODE
    # generate a sentence. A sentence starts with a <s> and ends with a </s>
    # Possiblly a use function is:
    #   np.random.choice(x, 1, p = y)
    # where x is a list of things to sample from
    # and y is a list of probability (of the same length as x)
    p = sentence[0]
    bigram = model["bigram"]
    
    while True:
        p = np.random.choice(list(bigram[p].keys()), 1, p = list(bigram[p].values()))[0]
        sentence.append(p)
        
        if p == "</s>":break
    return sentence


# In[44]:


# GENERATE A SENTENCE FROM THE MODEL

print("Generated sentence: ",generate(model))


# In[ ]:




