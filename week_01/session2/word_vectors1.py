import io, sys
import numpy as np
from heapq import *

def load_vectors(filename):
    fin = io.open(filename, 'r', encoding='utf-8', newline='\n')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.asarray(map(float, tokens[1:]))
    return data

## This function computes the cosine similarity between vectors u and v

def cosine(u, v):
    ## FILL CODE
    return u @ v/ (np.linalg.norm(u) * np.linalg.norm(v))

## This function returns the word corresponding to 
## nearest neighbor vector of x
## The list exclude_words can be used to exclude some
## words from the nearest neighbors search

def nearest_neighbor(x, word_vectors, exclude_words=[]):
    best_score = -1.0
    best_word = None

    ## FILL CODE
    for i in word_vectors:
        sim = cosine(x, word_vectors[i])
        if sim > best_score:
            best_score = sim
            best_word = i   

    return best_word

## This function return the words corresponding to the
## K nearest neighbors of vector x.
## You can use the functions heappush and heappop.

def knn(x, vectors, k):
    heap = []

    for i in vectors:
        dist = cosine(x, vectors[i])
        if len(heap) < k:
            heap.append((dist, i))
            
        else:
            m = min(heap, key= lambda x:x[0])
            idx_min = heap.index(m)
            if dist > m[0]:
                heap[idx_min] = dist, i      
    

    return sorted(heap, key= lambda x:x[0],reverse = True)
## This function return the word d, such that a:b and c:d
## verifies the same relation

def analogy(a, b, c, word_vectors):
    ## FILL CODE
    a = a.lower()
    b = b.lower()
    c = c.lower()  
    
    return nearest_neighbor((word_vectors[b] -word_vectors[a]  + word_vectors[c]), word_vectors)

## Compute the association strength between:
##   - a word w
##   - two sets of attributes A and B

def association_strength(w, A, B, vectors):
    strength = 0.0
    ## FILL CODE
    card_A = 1/len(A)
    card_B = 1/len(B)
    
    ass_A = card_A * np.sum([cosine(vectors[w], vectors[a]) for a in A])
    ass_B = card_B * np.sum([cosine(vectors[w], vectors[b]) for b in B])
    
    strength = ass_A - ass_B
    return strength

## Perform the word embedding association test between:
##   - two sets of words X and Y
##   - two sets of attributes A and B

def weat(X, Y, A, B, vectors):
    score = 0.0
    ## FILL CODE
    ass_x = np.sum([association_strength(x, A, B, vectors) for x in X])
    ass_y = np.sum([association_strength(y, A, B, vectors) for y in Y])
    score = ass_x - ass_y
    return score

######## MAIN ########

print('')
print(' ** Word vectors ** ')
print('')

word_vectors = load_vectors('wiki.en.vec')

print('similarity(apple, apples) = %.3f' %
      cosine(word_vectors['apple'], word_vectors['apples']))
print('similarity(apple, banana) = %.3f' %
      cosine(word_vectors['apple'], word_vectors['banana']))
print('similarity(apple, tiger) = %.3f' %
      cosine(word_vectors['apple'], word_vectors['tiger']))

print('')
print('The nearest neighbor of cat is: ' +
      nearest_neighbor(word_vectors['cat'], word_vectors))

knn_cat = knn(word_vectors['cat'], word_vectors, 5)
print('')
print('cat')
print('--------------')
for score, word in knn(word_vectors['cat'], word_vectors, 5):
    print (word + '\t%.3f' % score)

print('')
print('france - paris + rome = ' + analogy('paris', 'france', 'rome', word_vectors))

## A word about biases in word vectors:

print('')
print('similarity(genius, man) = %.3f' %
      cosine(word_vectors['man'], word_vectors['genius']))
print('similarity(genius, woman) = %.3f' %
      cosine(word_vectors['woman'], word_vectors['genius']))

## Replicate one of the experiments from:
##
## Semantics derived automatically from language corpora contain human-like biases
## Caliskan, Bryson, Narayanan (2017)

career = ['executive', 'management', 'professional', 'corporation', 
          'salary', 'office', 'business', 'career']
family = ['home', 'parents', 'children', 'family',
          'cousins', 'marriage', 'wedding', 'relatives']
male = ['john', 'paul', 'mike', 'kevin', 'steve', 'greg', 'jeff', 'bill']
female = ['amy', 'joan', 'lisa', 'sarah', 'diana', 'kate', 'ann', 'donna']

print('')
print('Word embedding association test: %.3f' %
      weat(career, family, male, female, word_vectors))
