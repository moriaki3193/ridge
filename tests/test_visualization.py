# -*- coding: utf-8 -*-
import pickle
import unittest
import numpy as np
import pandas as pd
from os import path
from scipy import sparse
from irmet.metrics import nDCG
from sklearn.metrics import mean_squared_error


BASEDIR = path.dirname(path.abspath(__file__))
PATH2TEMP = path.join(BASEDIR, 'tmp')


def build_filepath(k):
    filename = f'pred-k{k}-L24-iter20.npy'
    return path.join(PATH2TEMP, filename)


class TestVisualization(unittest.TestCase):

    def setUp(self):
        y_pred = {}
        # [START Import Evaluation Data]
        size = 0
        for k in range(2, 11):
            filepath = build_filepath(k)
            y_pred_k = np.load(filepath)
            size = len(y_pred_k)
            y_pred.update({f'FM{k}': y_pred_k})
        with open(path.join(PATH2TEMP, 'horseracing_raceids.pkl'), 'rb') as fp:
            raceids = pickle.load(fp)
            y_pred.update({'raceid': raceids[-size:]})
        self.data = pd.DataFrame(y_pred)
        # [END Import Evaluation Data]

    def test_visualize(self):
        print(self.data.head())


if __name__ == '__main__':
    unittest.main()
