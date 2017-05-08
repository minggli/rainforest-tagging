"""
label2vec

label2vec module produces label embeddings (Ul) as part of implementation of:

CNN-RNN: A Unified Framework for Multi-label Image Classification
Wang et al (2015)

This is done using downsized (1m) GloVe vectors via Python NLP framework: spaCy

GloVe: Global Vectors for Word Representation
Pennington et al (2014)
"""

import numpy as np
import spacy
import collections


class LabelVectorizer:

    def __init__(self):
        self._nlp = spacy.load('en_vectors_glove_md')

    @property
    def __corpus(self):
        return self._corpus

    @__corpus.setter
    def __corpus(self, value):
        if isinstance(value, collections.Iterable) and \
                not any(not isinstance(item, str) for item in value):
            self._corpus = value
            self._cleanse()
        else:
            raise TypeError('corpus must be a list-like iterable of strings.')

    def _cleanse(self):
        """replace non-ascii characters with a single space"""
        self._corpus = [''.join([i if (i.isalpha() or ord(i) == 39)
                        else ' ' for i in string]) for string in self._corpus]

    def fit(self, corpus):
        self.__corpus = corpus
        return self

    def transform(self):
        """output embedding of shape [vocal_size, n_dimensions]"""
        return np.array(
                [self._nlp(l).vector for l in self._corpus]).reshape(-1, 300)
