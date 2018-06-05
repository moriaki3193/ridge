# -*- coding: utf-8 -*-
import logging
import numpy as np
from tqdm import tqdm


logging.basicConfig(level=logging.DEBUG)


class FacMac:
    """2-way Factorization Machine.

    Parameters
    ----------
    k : int, the hyper-param of FM.
    b : float, the bias term.
    w : ndarray, the element-wise weight vector.
    V : ndarray, whose shape is (n_features, k).
        The weight matrix of pairwise interaction terms.
    l2 : float, L2 regularization term.
    eta : float, a.k.a learning rate.
    mode : str, 'original' or 'combination-dependent'.
        When 'original', the model is the one proposed by Rendle (2010).
        When 'combination', the model is the one preposed by Saigusa (2018).
    """
    
    def __init__(self, mode='original'):
        self.k = None
        self.b = None
        self.w = None
        self.V = None
        self.l2 = None
        self.eta = None
        self.p = None  # n_entities TODO docstringify
        self.q = None  # n_features TODO docstringify
        self.mode = mode

    def _orig_initialize_params(self, n_features, k, l2):
        self.k = k
        self.l2 = l2
        self.b = 0.0
        self.w = np.zeros(n_features)
        self.V = np.random.normal(scale=0.01, size=(n_features, k))
        # self.V = np.random.uniform(high=np.sqrt(k), size=(n_features, k))

    def _comb_initialize_params(self, p, q, k, l2):
        """Initialize parameters of Combination-Dependent FM.
        """
        n_features = 2 * p + q
        self.k = k
        self.l2 = l2
        self.b = 0.0
        self.w = np.zeros(n_features)
        self.V = np.random.normal(scale=0.01, size=(n_features, k))

    def _orig_update_params(self, x, residue):
        """Update parameters using Stochastic Gradient Descent method.
        """
        nonzero_ind = np.nonzero(x)[0]
        self.b = self.b + self.eta * (residue - self.l2 * self.b)
        self.w = self.w + self.eta * (residue * x - self.l2 * self.w)
        V_cur = self.V
        self.V -= self.l2 * self.V  # L2 Regularize in advance
        for f in range(self.k):
            shared_term = np.sum([V_cur[j, f] * x[j] for j in nonzero_ind])
            for i in nonzero_ind:
                x_i = x[i]
                V_if = V_cur[i, f]
                partial = x_i * shared_term - V_if * np.power(x_i, 2)
                self.V[i, f] = V_if + self.eta * residue * partial

    def _comb_update_params(self, x, residue):
        """Update parameters using Stochastic Gradient Descent method.
        """
        nonzero_ind = np.nonzero(x)[0]
        self.b = self.b + self.eta * (residue - self.l2 * self.b)
        self.w = self.w + self.eta * (residue * x - self.l2 * self.w)
        V_cur = self.V
        self.V -= self.l2 * self.V  # L2 Regularize in advance
        for f in range(self.k):
            shared_term = np.sum([V_cur[j, f] * x[j] for j in nonzero_ind])
            for i in nonzero_ind:
                if 0 <= i < p:
                    continue
                x_i = x[i]
                V_if = V_cur[i, f]
                partial = x_i * shared_term - V_if * np.power(x_i, 2)
                self.V[i, f] = V_if + self.eta * residue * partial

    def _orig_predict_one(self, x):
        """A prediction with a given feature vector.

        Utilizing Original Factorization Machine.

        Parameters
        ----------
        x : ndarray, whose shape is (n_features, )

        Return
        ------
        y_pred : float, a predicted target value.
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

    def _comb_predict_one(self, x):
        """A prediction with a given feature vector.

        Utilizing Combination-Dependent Factorization Machine.
        """
        nonzero_ind = np.nonzero(x)[0]
        pointwise_score = np.dot(self.w[nonzero_ind], x[nonzero_ind])
        pairwise_scores = []
        pairwise_noises = []  # Interaction among competitors latent vectors.
        for f in range(self.k):
            sum_squared = []
            squared_sum = []
            noises_sum_squared = []
            noises_squared_sum = []
            for i in nonzero_ind:
                x_i = x[i]
                V_if = self.V[i, f]
                term1 = V_if * x_i
                term2 = np.power(x_i, 2) * np.power(V_if, 2)
                sum_squared.append(term1)
                squared_sum.append(term2)
                if 0 <= i < self.p:
                    noises_sum_squared.append(term1)
                    noises_squared_sum.append(term2)
            sum_squared = np.power(np.sum(sum_squared), 2)
            squared_sum = np.sum(squared_sum)
            noises_sum_squared = np.power(np.sum(noises_sum_squared), 2)
            noises_squared_sum = np.sum(noises_squared_sum)
            pairwise_scores.append(sum_squared - squared_sum)
            pairwise_noises.append(noises_sum_squared - noises_squared_sum)
        pairwise_score = np.sum(pairwise_scores) - np.sum(pairwise_noises)
        return self.b + pointwise_score + 0.5 * pairwise_score

    def predict(self, X):
        """A prediction with given feature vectors.

        Parameters
        ----------
        X_test : ndarray, whose shape is (n_tests, n_features).

        Return
        ------
        y_pred : ndarray, whose shape is (n_tests, ).
        """
        n_samples, _ = X.shape
        y_pred = []

        predict_one = None
        if self.mode == 'original':
            predict_one = self._orig_predict_one
        elif self.mode == 'combination':
            predict_one = self._comb_predict_one

        for m in range(n_samples):
            x = X[m, :]
            t_pred = predict_one(x)
            y_pred.append(t_pred)

        return y_pred

    def fit(self, X_train, y_train, k=8, p=None, q=None,
            l2=0.02, eta=1e-3, n_iter=1000):
        n_samples, n_features = X_train.shape
        predict_one = None
        update_params = None
        if self.mode == 'original':
            self._orig_initialize_params(n_features, k, l2)
            predict_one = self._orig_predict_one
            update_params = self._orig_update_params
        elif self.mode == 'combination':
            self._comb_initialize_params(p, q, k, l2)
            predict_one = self._comb_predict_one
            update_params = self._comb_update_params

        # [START Fitting]
        self.eta = eta
        sample_indices = [i for i in range(n_samples)]
        for _epoch in tqdm(range(n_iter)):
            np.random.shuffle(sample_indices)
            for m in sample_indices:
                features = X_train[m]
                obs = y_train[m]
                pred = predict_one(features)
                update_params(features, obs - pred)
        # [END Fitting]

        return self
