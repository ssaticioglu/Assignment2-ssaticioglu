from __future__ import division
import scipy as sio
from scipy.sparse.csr import csr_matrix
import numpy as np

class MultinomialNB(object):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X = X.toarray()
        count_sample = X.shape[0]
        self.classes = [c for c in np.unique(y)]
        separated = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]
        self.class_log_prior_ = [np.log(len(i) / count_sample) for i in separated]
        count = np.array([np.array(i).sum(axis=0) for i in separated]) + self.alpha
        self.feature_log_prob_ = np.log(count / count.sum(axis=1)[np.newaxis].T)
        return self

    def predict_log_proba(self, X):
        return [(self.feature_log_prob_ * x).sum(axis=1) + self.class_log_prior_
                for x in X]

    def predict(self, X):
        X = X.toarray()
        temp = np.argmax(self.predict_log_proba(X), axis=1)
        toReturn = [self.classes[i] for i in temp]
        return toReturn