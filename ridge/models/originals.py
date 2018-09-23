# -*- coding: utf-8 -*-
"""Original Models module.

Implementations of my own proposed models.

Exapmle:
    >>> from ridge.models import CDFMRegressor
    >>> features, distance = CDFMRegressor.load_dataset(path1, path2)
    >>> model = CDFMRegressor(features, distance, k=8, eta=1e-3, n_iter=1000)
    >>> model.fit(l2_regularized=True)
    >>> model.predict(test_features, test_distance)

Todos:
    Cythonize load_dataset(), fit(), predict()
"""

import pickle
from typing import Dict, List, Tuple


SPLITTER = ' '   # split sections.
CONNECTOR = ':'  # map key-value in a section.
SEPARATOR = ','  # split values.


class CDFMRegressor:
    """Combination-Dependent Factorization Machines for regression tasks.

    Attributes:
        p (int): The number of unique entities in dataset.

        q (int): The number of dimensions on feature set.

        u (List[float]): Weights on distance vectors.

        w (List[float]): Pointwise weights on feature vectors.

        Ve (List[List[float]]): Latent vectors of entity indexing vectors
            on pairwise interaction terms. The number of dimensions of
            each vector is k.

        Vc (List[List[float]]): Latent vectors of combination indexing vectors
            on pairwise interaction terms. The number of dimensions of
            each vector is k.

        Ve (List[List[float]]): Latent vectors of context vectors
            on pairwise interaction terms. The number of dimensions of
            each vector is k.

        k (int): The hyper-parameter of this model, which represents the
            number of dimensions of Vc, Ve and Vf. Defaults to 10.

        eta (float): A.k.a learning rate. Defaults to 1e-2.

        n_iter (int): The maximum number of iterations. Defaults to 1000.

    References:
        http://db-event.jpn.org/deim2018/data/papers/297.pdf
    """

    @property
    def p(self):
        return self.p

    @property
    def q(self):
        return self.q

    @property
    def n_iter(self):
        return self.n_iter

    @property
    def k(self):
        return self.k

    @staticmethod
    def load_dataset(
            features_path: str,
            distance_path: str = None) -> tuple:
        """Load dataset from specified file paths.
        """
        features = {}
        with open(features_path, mode='r') as fp:
            for line in fp.readlines():
                # Parse a line
                t, gid, eid, cid, feat, _annot = line.split(SPLITTER)
                t = float(t)
                _, gid = gid.split(CONNECTOR)
                _, eid = eid.split(CONNECTOR)
                _, cid = cid.split(CONNECTOR)
                cid = cid.split(SEPARATOR)
                _, feat = feat.split(CONNECTOR)
                feat = [float(elem) for elem in feat.split(SEPARATOR)]

                # Insert g(roup)id if not exist.
                if gid not in features:
                    features.update({gid: []})

                # Update features. TODO faster!!! cythonize maybe...
                features[gid].append({
                        'target': t,
                        'eid': eid,
                        'cid': cid,
                        'features': feat})

        distance = None
        if distance_path is not None:
            with open(distance_path, mode='r') as fp:
                for line in fp.readlines():
                    pass

        return features, distance

    def __init__(
            self,
            features = None,
            distance = None,
            k: int = 10,
            eta: float = 1e-2,
            n_iter: int = 1000) -> None:
        """Initialization of an instance.

        Args:
            features: pass

            distance: pass
        """
        self.features = features
        self.distance = distance

        self.p, self.q = self._extract_n_dimensions()

        self.u: List[float] = None
        self.w: List[float] = None
        self.Ve: List[List[float]] = None
        self.Vc: List[List[float]] = None
        self.Vf: List[List[float]] = None

        self.k: int = k
        self.eta: float = eta
        self.n_iter: int = n_iter

    def _extract_n_dimensions(self) -> Tuple[int]:
        """Extract p & q, which are the number of unique entites & features.
        """
        p, q = 1, 1
        return p, q

    def save(self, path: str) -> None:
        """Saving this model.
        """
        with open(path, mode='wb') as fp:
            pickle.dump(self, fp)

    def fit(self, l2_regularized=True) -> None:
        """Fitting the model parameters.

        Args:
            l2_regularized (bool): L2-regularize the loss function or not.
                Defaults to True.
        """
        for m in range(self.n_iter):
            pass

    def predict(self):
        pass
