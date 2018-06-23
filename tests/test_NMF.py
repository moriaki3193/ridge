# -*- coding: utf-8 -*-
import unittest
import numpy as np
from ridge.models import NNMatFac


class TestNNMatFac(unittest.TestCase):

    def setUp(self):
        """Set up this test suite.

        Variables
        ---------
        data : np.ndarray, whose shape is (n_users, n_items).
        """
        self.data = np.array([
            [5, 3, 0, 1],
            [4, 0, 0, 1],
            [1, 1, 0, 5],
            [1, 0, 0, 4],
            [0, 1, 5, 4],
        ])

    def test_nonnegative_matrix_factorization(self):
        V = self.data.T
        M = 3
        model = NNMatFac().fit(V, M, n_iter=100)
        print('KL devergence')
        print(model.W)
        print(model.H)
        print(np.round(np.dot(model.W, model.H), 2))
        print(V)


if __name__ == '__main__':
    unittest.main()
