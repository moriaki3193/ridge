# -*- coding: utf-8 -*-
import logging
import numpy as np
from tqdm import tqdm


logging.basicConfig(level=logging.DEBUG)


class FacMac:
    
    def __init__(self):
        self.epoch = None
        self.k = None
        self.b = None
        self.w = None
        self.V = None
        self.eta = None

    def _initialize_params(self, n_features, k):
        self.k = k
        self.b = 0.0
        self.w = np.zeros(n_features)
        # self.V = np.random.uniform(high=np.sqrt(k), size=(n_features, k))
        self.V = np.random.normal(scale=0.01, size=(n_features, k))

    def _update_params(self, x, residue):
        """Update parameters using Stochastic Gradient Descent method.
        """
        nonzero_ind = np.nonzero(x)[0]
        self.b = self.b + 2 * self.eta * residue
        self.w = self.w + 2 * self.eta * residue * x
        V_cur = self.V
        for f in range(self.k):
            shared_term = np.sum([V_cur[j, f] * x[j] for j in nonzero_ind])
            for i in nonzero_ind:
                x_i = x[i]
                V_if = V_cur[i, f]
                partial = x_i * shared_term - V_if * np.power(x_i, 2)
                self.V[i, f] = V_if + 2 * self.eta * residue * partial

    def predict_one(self, x):
        """A prediction with a given feature vector.
        """
        nonzero_ind = np.nonzero(x)[0]
        pointwise_score = np.dot(self.w[nonzero_ind], x[nonzero_ind])
        pairwise_scores = []
        for f in range(self.k):
            sum_squared = []
            squared_sum = []
            for i in nonzero_ind:
                x_i = x[i]
                V_if = self.V[i, f]
                sum_squared.append(V_if * x_i)
                squared_sum.append(np.power(x_i, 2) * np.power(V_if, 2))
            sum_squared = np.power(np.sum(sum_squared), 2)
            squared_sum = np.sum(squared_sum)
            pairwise_scores.append(sum_squared - squared_sum)
        return self.b + pointwise_score + 0.5 * np.sum(pairwise_scores)

    def fit(self, X_train, y_train, k=8, eta=1e-3, n_iter=1000):
        n_samples, n_features = X_train.shape
        self.eta = eta
        self._initialize_params(n_features, k)
        for _epoch in tqdm(range(n_iter)):
            self.epoch = _epoch  # TODO 外す
            for m in range(n_samples):
                features = X_train[m]
                obs = y_train[m]
                pred = self.predict_one(features)
                self._update_params(features, obs - pred)
        return self
