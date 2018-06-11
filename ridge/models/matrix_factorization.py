# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm
from numpy.linalg import norm

# from ridge.src import gradient_steps
import gradient_steps


class MatFac:
    """Matrix Factorization
    """

    def __init__(self):
        self.P = None
        self.Q = None
        self.loss_series = []

    def _calc_loss(self, errors, l2):
        """エポックごとの損失を計算する

        params
        ------
        errors : list of float
                A difference between observed and predicted values.

        variables
        ---------
        squared_error_sum : float 実測値と予測値の二乗誤差の総和
        reg_term : L2正則化項
        """
        squared_error_sum = sum([residue ** 2 for residue in errors])
        reg_term = (l2 / 2.0) * (norm(self.P) + norm(self.Q))
        loss = squared_error_sum + reg_term
        self.loss_series.append(loss)
        return loss

    def fit(self, ratings, k=8, n_iter=1000,
            alpha=0.0005, l2=0.02, threshold=0.001, verbose=True):
        """レーティングを受け取り，ユーザとアイテムの`k'次元の潜在ベクトルを獲得する

        params
        ------
        ratings : ndarray or matrix, whose shape is (n_users, n_items)
                A matrix expressing the ratings of items by users.
        k       : int
                The number of dimensions of latent vectors.
        n_iter  : int
                The number of iterations when fitting.
        alpha   : float
        l2      : float
                A L2 regularization term.
        threshold : float
        verbose : bool
                Whether it displays progress bar when fitting.
        """
        ratings = np.asarray(ratings)

        # [START Initialize latent matrices]
        n_users, n_items = ratings.shape
        mu = 0.0
        sigma = 0.01
        self.P = np.random.normal(mu, sigma, (k, n_users))
        self.Q = np.random.normal(mu, sigma, (k, n_items))
        # [END Initialize latent matrices]

        # [START Fitting]
        pbar = tqdm(total=n_iter) if verbose else None
        for _epoch in range(n_iter):
            self.P, self.Q, errors = gradient_steps.mf(ratings, self.P, self.Q, alpha)
            loss = self._calc_loss(errors, l2)
            if verbose:
                pbar.update(1)
            if loss < threshold:
                break
        if verbose:
            pbar.close()
        # [END Fitting]

        return self

    def predict_one(self, user_id, item_id):
        """self.fitの結果を利用して，ユーザのアイテムへのレーティングの予測値を出力する
        """
        p = self.P[:, user_id]
        q = self.Q[:, item_id]
        return np.dot(p, q)
