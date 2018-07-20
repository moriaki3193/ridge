# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm
from scipy import sparse
from ridge.racer import (
    predictors,
    link_functions,
    loss_calculators,
)


class FMRegressor:
    """2-way Factorization Machine.

    Attributes
    ----------
    k : int
        The hyper-param of FM.
    b : float
        The bias term.
    w : ndarray
        The element-wise weight vector.
    V : ndarray, whose shape is (n_features, k).
        The weight matrix of pairwise interaction terms.
    l2 : float
        L2 regularization term.
    eta : float
        a.k.a learning rate.
    """

    def __init__(self):
        self.k = None
        self.b = None
        self.w = None
        self.V = None
        self.l2 = None
        self.eta = None

    def _extract_feature_row(self, X, m):
        if isinstance(X, np.matrix):
            return np.asarray(X)[m]
        if isinstance(X, sparse.csr_matrix):
            return np.squeeze(np.asarray(X[m].todense()))
        elif isinstance(X, np.ndarray):
            return X[m]

    def _initialize_params(self, n_features, k, l2):
        self.k = k
        self.l2 = l2
        self.b = 0.0
        self.w = np.zeros(n_features)
        self.V = np.random.normal(scale=1e-2, size=(n_features, k))

    def _update_params(self, x: np.ndarray, error: float):
        """Update parameters using Stochastic Gradient Descent method.

        Paramters
        ---------
        x : np.ndarray
            A feature vector of an instance in training dataset.
        error: float
            A prediction error between observed and predicted values.
        """
        nonzero_ind = np.nonzero(x)[0]
        self.b = self.b + self.eta * (error - self.l2 * self.b)
        self.w = self.w + self.eta * (error * x - self.l2 * self.w)
        V_cur = self.V
        self.V -= self.l2 * self.V  # L2 Regularize in advance
        for f in range(self.k):
            shared_term = np.sum([V_cur[j, f] * x[j] for j in nonzero_ind])
            for i in nonzero_ind:
                x_i = x[i]
                V_if = V_cur[i, f]
                partial = x_i * shared_term - V_if * np.power(x_i, 2)
                self.V[i, f] = V_if + self.eta * error * partial

    def __score(self, x):
        """A prediction with a given feature vector.

        Utilizing Original Factorization Machine.

        Parameters
        ----------
        x : ndarray, whose shape is (n_features, )

        Return
        ------
        y_pred : float, a predicted target value.
        """
        return predictors.fm(self.b, self.w, self.V, x)

    def predict(self, X_test):
        """A prediction with given feature vectors.

        Parameters
        ----------
        X_test : ndarray, whose shape is (n_tests, n_features).

        Return
        ------
        y_pred : ndarray, whose shape is (n_tests, ).
        """
        n_samples, _ = X_test.shape
        y_pred = []

        # [START Predict one by one]
        for m in range(n_samples):
            x = self._extract_feature_row(X_test, m)
            this_y_pred = self.__score(x)
            y_pred.append(this_y_pred)
        # [END Predict one by one]

        return np.array(y_pred)

    def fit(self, X_train, y_train, k=8, l2=0.02, eta=1e-3, n_iter=1000):
        """Parameter fitting.

        Parameters
        ----------
        X_train : sparse.csr_matrix, np.ndarray or np.matrix
            Its shape is (n_samples, n_features).
        y_train : np.ndarray
            Its shape is (n_samples, ).
        n_iter : int
            The maximum number of iteration.
        """
        n_samples, n_features = X_train.shape
        self._initialize_params(n_features, k, l2)
        # [START Fitting]
        self.eta = eta
        sample_indices = [i for i in range(n_samples)]
        for _epoch in tqdm(range(n_iter)):
            np.random.shuffle(sample_indices)
            for m in sample_indices:
                x = self._extract_feature_row(X_train, m)
                obs = y_train[m]
                pred = self.__score(x)
                self._update_params(x, obs - pred)
        # [END Fitting]

        return self


##############################################################################


class FMClassifier:
    """Factorization Machines for Classification.

    Features of this model.
    -----------------------
    + L2 Regularization.
    + Probability prediction.
    + Memoize a series of loss.
    + (Locally) Optimized logistic loss.
    + Probabilistic Approximation of learning rate (a.k.a Robbins-Monro).
    """

    def __init__(self):
        self.k = None
        self.b = None
        self.w = None
        self.V = None
        self.l2 = None
        self.eta = None
        self.type_X = None
        # self.loss_series = None

    def _initialize_loss_series(self, n_iter):
        """Initialize a series of loss when fitting.
        """
        self.loss_series = np.zeros(n_iter)

    def _update_loss_series(self, epoch, X, y):
        """Append Cross Entropy Loss to `loss_series'.
        """
        loss = loss_calculators.fm_cross_entropy(self.b, self.w, self.V, X, y)
        self.loss_series[epoch] = loss

    def _initialize_params(self, n_features, k, l2, eta):
        """Initialize parameters of this model.
        """
        self.k = k
        self.b = 0.0
        self.w = np.zeros(n_features)
        self.V = np.random.normal(scale=1e-2, size=(n_features, k))
        self.l2 = l2
        self.eta = eta

    def _update_params(self, x, y):
        """Update parameters of this model.

        Employing Stochastic Gradient Descent algorithm
        with Robbins-Monro method.

        Parameters
        ----------
        x : np.ndarray, whose shape is (n_features, )
        y : int {0, +1}

        Variables
        ---------
        V_cur : Current values of latent matrix V.
        """
        x = self.__extract_feature_row(x)
        z = self._predict_proba(x)
        coef = z - y
        self.b -= self.eta * (coef + self.l2 * self.b)
        self.w -= self.eta * (coef * x + self.l2 * self.w)
        Vt = self.V.T
        for f in np.arange(start=0, stop=self.k):
            Vt_f = Vt[f, :]
            nnz_ind = np.nonzero(x)[0]
            shared_term = np.sum([Vt_f[j] * x[j] for j in nnz_ind])
            for i in nnz_ind:
                first_term = x[i] * shared_term
                second_term = Vt_f[i] * np.square(x[i])
                nabra = coef * (first_term - second_term)
                self.V[i, f] -= self.eta * (nabra + self.l2 * Vt_f[i])

    def __score(self, x):
        """Scoring with a given feature `x'.

        Parameters
        ----------
        x : np.ndarray, whose shape is (n, ).

        Return
        ------
        score : float, regression score by FMs.
        """
        return predictors.fm(self.b, self.w, self.V, x)

    def __extract_feature_row(self, x):
        if self.type_X == sparse.csr_matrix:
            return np.squeeze(np.asarray(x.todense()))
        elif self.type_X == np.ndarray:
            return x

    def _predict_proba(self, x):
        """Predict probability with a given feature `x'.

        A.k.a Sigmoid function.

        Parameters
        ----------
        x : np.ndarray, whose shape is (n_features, ).
        """
        return link_functions.sigmoid(self.__score(x))

    def fit(self, X, y, k=8, l2=1e-2, eta=1e-2, n_iter=1000, verbose=True):
        """Fitting parameters of this model.

        Parameters
        ----------
        X       : np.ndarray, whose shape is (n_samples, n_features).
        y       : np.ndarray, whose shape is (n_samples, ).
                  Each element in y must be either {0, +1}.
        k       : int, a hyper-param of this model. Default to 8.
        l2      : float, a L2 Regularization term. Default to 1e-2.
        eta     : float, an initial value of learning rate.
                  This eta is attenuated as the number of iteration increases.
        verbose : bool, whether display a progress bar of iteration.
        """
        n_samples, n_features = X.shape
        self.type_X = type(X)
        self._initialize_params(n_features, k, l2, eta)
        # self._initialize_loss_series(n_iter)
        sample_indices = np.arange(start=0, stop=n_samples)
        if verbose:
            pbar = tqdm(total=n_iter)
            pbar.set_description(f'Fitting')
        for epoch in np.arange(start=0, stop=n_iter):
            np.random.shuffle(sample_indices)
            for m in sample_indices:
                row = X[m]
                obs = y[m]
                self._update_params(row, obs)
            self.eta = eta / (epoch + 1)
            # self._update_loss_series(epoch, X, y)
            if verbose:
                pbar.update(1)
        pbar.close()
        return self

    def predict(self, X, target='0-1'):
        """Prediction of this model.

        Parameters
        ----------
        X : np.ndarray, whose shape is (n_samples, n_features).

        Return
        ------
        y_pred : np.ndarray, whose shape is (n_samples, ).
                 if `target' is 'probability', return estimated probabilities.
                 otherwise, return a series of {0, +1} values.
        """
        self.type_X = type(X)
        t_pred = []
        for x in X:
            x = self.__extract_feature_row(x)
            t_pred.append(self._predict_proba(x))
        if target == 'probability':
            return np.array(t_pred)
        else:
            probas = np.array(t_pred)
            return np.where(probas >= 0.5, 1, 0)


##############################################################################
