import numpy as np
import fasttext
import os
import warnings
from gensim.models import Word2Vec
warnings.filterwarnings('ignore')
import torch
from nltk import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from string import punctuation
import utils.consts as consts
def get_vect(word, model):
    try:
        return model.wv[word]
    except KeyError:
        return np.zeros((model.vector_size,))

def sum_vectors(phrase, model):
    return sum(get_vect(w, model) for w in phrase)

def word2vec_features(X, model):
    feats = np.vstack([sum_vectors(p, model) for p in X])
    return feats


def preprocess_sentence(text):
    stop_words = set(stopwords.words('english'))
    text = text.replace('/', ' or ')
    text = text.replace('.-', ' .- ')
    #     text = text.replace('.', ' . ')
    text = text.lower()

    tokens = [token for token in word_tokenize(text) if token not in punctuation and token not in stop_words]

    return ' '.join(tokens)
def fasttext_encoding(x_train,x_test,ngrams, embedding_size):
    X_train = []
    for e in x_train:
        tokenized_words = preprocess_sentence(e)
        X_train.append(tokenized_words)
    X_test = []
    for e in x_test:
        tokenized_words = preprocess_sentence(e)
        X_test.append(tokenized_words)

    with open(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_TEXT,'train.txt'), 'w', encoding='utf-8') as f:
        for rec in X_train:
            f.write(str(rec) + '\n')
    model = fasttext.train_unsupervised(os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_TEXT,'train.txt'), wordNgrams=ngrams, dim=embedding_size)
    x_train = []

    for e in X_train:
        x_train.append(model.get_sentence_vector(str(e)))
    x_test = []
    for e in X_test:
        x_test.append(model.get_sentence_vector(str(e)))
    return x_train,x_test
def w2vec_encoding(x_train,x_test,embeddingsize,seed_value):
    X_train = []
    for e in x_train:
        tokenized_words = preprocess_sentence(e)
        X_train.append(tokenized_words.split())
    X_test = []
    for e in x_test:
        tokenized_words = preprocess_sentence(e)
        X_test.append(tokenized_words.split())

    model_w2v = Word2Vec(min_count=2,
                         # sentences=tokenized_words,
                         seed=seed_value,
                         window=10,
                         vector_size=embeddingsize)

    model_w2v.build_vocab(X_train)
    model_w2v.train(X_train, total_examples=len(tokenized_words), epochs=100)
    x_train = word2vec_features(X_train, model_w2v)
    x_test = word2vec_features(X_test, model_w2v)
    return x_train,x_test