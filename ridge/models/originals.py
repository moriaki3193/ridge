# -*- coding: utf-8 -*-
"""Original Models module.

Implementations of my own proposed models.
"""
import pickle
from typing import Dict, List, Tuple


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
            number of dimensions of Vc, Ve and Vf.

        n_iter (int): The maximum number of iterations.

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

    def __init__(
            self,
            features,
            distance,
            k: int = 10,
            n_iter: int = 1000) -> None:
        """Initialization of an instance.

        Args:
            features: pass

            distance: pass
        """
        self.features = features
        self.distance = distance

        self.p, self.q = self.__extract_n_dimensions()

        self.u: List[float] = None
        self.w: List[float] = None
        self.Ve: List[List[float]] = None
        self.Vc: List[List[float]] = None
        self.Vf: List[List[float]] = None

        self.k: int = k
        self.n_iter: int = n_iter

    def __extract_n_dimensions(self) -> Tuple[int]:
        """Extract p & q, which are the number of unique entites & features.
        """
        p, q = 1, 1
        return p, q

    def save(self, path: str) -> None:
        """Saving this model.
        """
        with open(path, mode='wb') as fp:
            pickle.dump(self, fp)

    def fit(self):
        """Fitting the model parameters.
        """
        for m in range(self.n_iter):
            pass

    def predict(self):
        pass
