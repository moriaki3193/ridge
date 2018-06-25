# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm
from scipy import sparse
from ridge.racer import loss_calculators


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
        self.loss_series = None

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
        z = self._predict_proba(x)
        coef = z - y
        self.b -= self.eta * (coef + self.l2 * self.b)
        self.w -= self.eta * (coef * self.w + self.l2 * self.w)
        V_cur = self.V
        for f in np.arange(start=0, stop=self.k):  # TODO 列方向の取り出しは低速なので改良する
            nnz_ind = np.nonzero(x)[0]
            shared_term = np.sum([V_cur[j,f] * x[j] for j in nnz_ind])
            for i in nnz_ind:
                first_term = x[i] * shared_term
                second_term = V_cur[i,f] * np.square(x[i])
                self.V[i,f] -= self.eta * (coef * (first_term - second_term) + self.l2 * V_cur[i,f])

    def __score(self, x):
        """Scoring with a given feature `x'.

        Parameters
        ----------
        x : np.ndarray, whose shape is (n, ).

        Return
        ------
        score : float, regression score by FMs.
        """
        nonzero_ind = np.nonzero(x)[0]
        pointwise_score = np.dot(self.w[nonzero_ind], x[nonzero_ind])
        pairwise_scores = []
        for f in np.arange(start=0, stop=self.k):
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

    def _predict_proba(self, x):
        """Predict probability with a given feature `x'.

        A.k.a Sigmoid function.

        Parameters
        ----------
        x : np.ndarray, whose shape is (n_features, ).
        """
        return 1 / (1 + np.exp(-self.__score(x)))

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
        self._initialize_params(n_features, k, l2, eta)
        self._initialize_loss_series(n_iter)
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
            self._update_loss_series(epoch, X, y)
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
        if target == 'probability':
            return np.array([self._predict_proba(x) for x in X])
        else:
            probas = np.array([self._predict_proba(x) for x in X])
            return np.where(probas >= 0.5, 1, 0)


##############################################################################


class FMRegressor:
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
    """
    
    def __init__(self, type_X=np.ndarray):
        self.type_X = type_X
        self.k = None
        self.b = None
        self.w = None
        self.V = None
        self.l2 = None
        self.eta = None

    def _extract_feature_row(self, X, m):
        if self.type_X == sparse.csr_matrix:
            return np.squeeze(np.asarray(X[m].todense()))
        elif self.type_X == np.ndarray:
            return X[m]

    def _initialize_params(self, n_features, k, l2):
        self.k = k
        self.l2 = l2
        self.b = 0.0
        self.w = np.zeros(n_features)
        self.V = np.random.normal(scale=1e-2, size=(n_features, k))

    def _update_params(self, x, residue):
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

    def _predict_one(self, x):
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
            this_y_pred = self._predict_one(x)
            y_pred.append(this_y_pred)
        # [END Predict one by one]

        return np.array(y_pred)

    def fit(self, X_train, y_train, k=8, l2=0.02, eta=1e-3, n_iter=1000):
        """Parameter fitting.

        Parameters
        ----------
        X_train : sparse.csr_matrix or ndarray,
                  whose shape is (n_samples, n_features).
        y_train : ndarray, whose shape is (n_samples, ).
        n_iter  : The maximum number of iteration.
        """
        n_samples, n_features = X_train.shape
        self._initialize_params(n_features, k, l2)
        # [START Fitting]
        self.eta = eta
        sample_indices = [i for i in range(n_samples)]
        for _epoch in tqdm(range(n_iter)):
            np.random.shuffle(sample_indices)
            for m in sample_indices:
                features = self._extract_feature_row(X_train, m)
                obs = y_train[m]
                pred = self._predict_one(features)
                self._update_params(features, obs - pred)
        # [END Fitting]

        return self


##############################################################################


class FaFacMac:
    """Field-aware Factorization Machine.

    Parameters
    ----------
    k : int, a hyper-param of Factorization Machines.
    V : dict of np.array, whose key is the field,
        and shape of its value is (n_features, k).
    """

    def __init__(self):
        self.k = None
        self.V = None
        self.G = None
        self.l2 = None
        self.eta = None

    def _inspect_structure(self, X):
        """Inspect dataset X and extract its shape.

        Return
        ------
        shape : tuple of int, n_samples, n_features, n_fields.
        """
        n_samples, n_features, n_fields = (0, 0, 0)
        if type(X) == np.ndarray and X.ndim == 3:
            n_samples, n_features, n_fields = X.shape
        elif type(X) == list or type(X) == np.ndarray:
            n_samples = len(X)
            n_features, n_fields = X[0].shape
        return (n_samples, n_features, n_fields)

    def _initialize_params(self, n_features, n_fields, k, l2, eta):
        """Initializing the parameters.
        """
        self.k = k
        self.l2 = l2
        self.eta = eta
        self.V = {field_id: np.random.normal(scale=1e-2, size=(n_features, k))
                  for field_id in range(n_fields)}
        self.G = np.ones((n_features, n_fields, k))

    def _predict_one(self, x):
        """Prediction with a given feature.

        Parameters
        ----------
        x : np.array OR sparse.coo_matrix
            whose shape are (n_features, n_fields).
        """
        if type(x) == sparse.coo_matrix:
            x = x.toarray()

    def _update_params(self, x, obs, pred):
        if type(x) == sparse.coo_matrix:
            x = x.toarray()

    def fit(self, X_train, y_train, k=8,
            l2=0.02, eta=1e-2, n_iter=1000):
        """Fitting using the given X_train & y_train.

        Employing AdaGrad so as to fit efficiently.

        Parameters
        ----------
        X_train : np.array or array-like of sparse.coo_matrix.
            Its shape is (n_samples, n_features, n_fields).
        y_train : np.array, whose shape is (n_samples, ).
            Each value in y_train has to be 0 or 1.
        k : int, a hyper-param of Factorization Machine.
            Default to 8.
        l2 : float, L2 regularization term.
            Default to 0.02.
        eta : float, initial learning rate of AdaGrad.
            Default to 1e-2 (0.01).
        n_iter : int, the number of iteration. A.k.a epochs.
            Default to 1000.
        """
        n_samples, n_features, n_fields = self._inspect_structure(X_train)
        self._initialize_params(n_features, n_fields, k, l2, eta)

        # [START Fitting]
        sample_indices = [_i for _i in range(n_samples)]
        for _epoch in tqdm(range(n_iter)):
            np.random.shuffle(sample_indices)  # shuffle
            for m in sample_indices:
                x = X_train[m]
                obs = y_train[m]
                pred = self._predict_one(x)
                self._update_params(x, obs, pred)
        # [END Fitting]

        return self


class CDFacMac:
    """Combination-Dependent Factorization Machine
    """

    def __init__(self, type_X=np.ndarray):
        self.type_X = type_X
        self.n_entities = None
        self.k = None
        self.b = None
        self.w = None
        self.V = None
        self.l2 = None
        self.eta = None

    def _extract_feature_row(self, X, m):
        if self.type_X == sparse.csr_matrix:
            return np.squeeze(np.asarray(X[m].todense()))
        elif self.type_X == np.ndarray:
            return X[m]

    def _initialize_params(self, n_entities, n_features, k, l2):
        self.n_entities = n_entities
        self.k = k
        self.l2 = l2
        self.b = 0.0
        self.w = np.zeros(n_features)
        self.V = np.random.normal(scale=1e-2, size=(n_features, k))

    def _update_params(self, x, residue):
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

    def _predict_one(self, x):
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
        # [START Revome Noises]
        pairwise_noises = [0.0]
        for idx, i in enumerate(nonzero_ind):
            if idx >= len(nonzero_ind) - 1:
                break
            j = nonzero_ind[idx + 1]
            if j < self.n_entities:
                pairwise_noises.append(np.dot(self.V[i], self.V[j]))
        # [END Revome Noises]
        return self.b + pointwise_score + 0.5 * np.sum(pairwise_scores) - np.sum(pairwise_noises)

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
            this_y_pred = self._predict_one(x)
            y_pred.append(this_y_pred)
        # [END Predict one by one]

        return np.array(y_pred)

    def fit(self, X_train, y_train, n_entities,
            k=8, l2=0.02, eta=1e-3, n_iter=1000):
        """Parameter fitting.

        Parameters
        ----------
        X_train : sparse.csr_matrix or ndarray,
                  whose shape is (n_samples, n_features).
        y_train : ndarray, whose shape is (n_samples, ).
        n_iter  : int, the maximum number of iteration.
        n_entities : int, the number of entities.
        """
        n_samples, n_features = X_train.shape
        self._initialize_params(n_entities, n_features, k, l2)
        # [START Fitting]
        self.eta = eta
        sample_indices = [i for i in range(n_samples)]
        for _epoch in tqdm(range(n_iter)):
            np.random.shuffle(sample_indices)
            for m in sample_indices:
                features = self._extract_feature_row(X_train, m)
                obs = y_train[m]
                pred = self._predict_one(features)
                self._update_params(features, obs - pred)
        # [END Fitting]

        return self
