# -*- coding: utf-8 -*-
import numpy as np
from typing import List


Dataset = List[np.matrix]


class EmptyDatasetError(Exception):
    pass


class IllegalInputError(Exception):
    pass


def _validate_input(X: Dataset, w: np.ndarray) -> bool:
    """入力されるデータセットの形式が正しいかどうか検証する．
    """
    if len(X) == 0:
        raise EmptyDatasetError('The given dataset is empty.')
    elif X[0].ndim != 2:
        raise IllegalInputError('Each sample must be represented as a 2d matrix.')
    elif len(X) != len(w):
        raise IllegalInputError('The lengths of X and w must be same.')
    return True


class ConditionalLogit:
    """Conditional Logit Model

    Most Likelihood Estimation by Gradient Descend Method.
    """

    @classmethod
    def const_X_ast(cls, X: Dataset, w: np.ndarray) -> np.matrix:
        """各事例で選ばれた選択肢の特徴量を束ねた行列を作成する．
        """
        # Validation Step
        _validate_input(X, w)
        # Construction Step
        n_samples = len(X)
        _, n_features = X[0].shape
        X_ast = np.asmatrix(np.empty((n_samples, n_features)))
        for i, (X_i, w_i) in enumerate(zip(X, w)):
            X_ast[i, :] = X_i[w_i]
        return X_ast

    @classmethod
    def add_constant(cls, X: Dataset) -> Dataset:
        return [np.hstack((np.ones(X_i.shape[0]), X_i)) for X_i in X]

    def __init__(self, use_bias: bool, tol: float = 1e-4, max_iter: int = 1000, eta: float = 1e-2):
        """Initialize an instance.

        Parameters
        ----------
        use_bias : 説明変数に定数項を追加するかどうか．
        tol : パラメータ更新の終了条件となる閾値．
        max_iter : パラメータ推定時の最大反復回数．
        eta : 勾配降下法の学習率．
        """
        self.use_bias = use_bias
        self.tol = tol
        self.max_iter = max_iter
        self.eta = eta
        self.beta = np.array([0.0]) if self.use_bias else np.array([])
        self.log_likelihood = np.array([])

    def _initialize_beta(self, n_features: int):
        """選択肢の特徴量の次元数に応じて推定するパラメータbetaの初期化を行う．
        """
        self.beta = np.append(self.beta, np.zeros(n_features))

    def _calc_P_ast(self, X: Dataset, w: np.array) -> np.ndarray:
        """各事例の選ばれた選択肢の選択される確率を求める
        """
        P_ast = np.empty(len(w))
        for i, (X_i, w_i) in enumerate(zip(X, w)):
            estimated_probas: np.ndarray = np.exp(np.ravel(np.dot(X_i, self.beta)))
            P_ast[i] = estimated_probas[w_i] / estimated_probas.sum()
        return P_ast

    def _1_order_diff_coef(self, X: Dataset, w: np.ndarray) -> (np.ndarray, np.ndarray):
        """データセットXについての対数尤度関数のself.betaについての1階微分係数を返す．

        Parameters
        ----------
        X : Shape is (n_samples, n_choices, n_variables). Training dataset matrix.
        w : Shape is (n_samples, ). Indicates what was chosen in each samples.

        Return
        ------
        coef : Shape is (n_variables). 1-order differential coefficient vector.
        """
        # Validation
        # 各事例で選ばれた選択肢の特徴量を束ねた行列を作る
        X_ast: np.matrix = self.const_X_ast(X, w)
        P_ast: np.ndarray = self._calc_P_ast(X, w)
        return X_ast.dot(1.0 - P_ast), P_ast

    def estimate(self, X: Dataset, w: np.ndarray):
        """Estimation of the parameters.
        """
        # Validation Step
        _validate_input(X, w)
        # Initialize parameters
        _, n_features = X[0].shape
        self._initialize_beta(n_features)
        # Transform the Input
        X = self.add_constant(X) if self.use_bias else X
        # Estimation of the parameters.
        n_iter = 0
        while n_iter < self.max_iter:
            np.random.shuffle(X)
            nabra, P_ast = self._1_order_diff_coef(X, w)
            self.beta = self.beta - self.eta * nabra
            self.log_likelihood = np.append(self.log_likelihood, np.sum(np.log(P_ast)))
            n_iter += 1

    def predict(self, X: Dataset):
        # Transform the Input
        X = self.add_constant(X) if self.use_bias else X
