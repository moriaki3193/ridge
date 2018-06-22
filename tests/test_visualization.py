# -*- coding: utf-8 -*-
import pickle
import unittest
import numpy as np
import pandas as pd
from os import path
from tqdm import tqdm
from scipy import sparse
from pprint import pprint
from irmet.metrics import nDCG
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error


GROUP_KEY = 'raceid'
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
        for k in range(2, 5):
            filepath = path.join(PATH2TEMP, f'pred-k{k}-L24-iter20-CDFM.npy')
            y_pred_k = np.load(filepath)
            y_pred.update({f'CDFM{k}': y_pred_k})
        with open(path.join(PATH2TEMP, 'horseracing_raceids.pkl'), 'rb') as fp:
            raceids = pickle.load(fp)
            y_pred.update({GROUP_KEY: raceids[-size:]})
        y_obs = np.load(path.join(PATH2TEMP, 'horseracing_targets.npy'))
        y_pred.update({'obs': y_obs[-size:]})
        # [END Import Evaluation Data]
        self.data = pd.DataFrame(y_pred)

    def test_visualize(self):
        rankrel_map = {
            0.0: 5.0, 1.0: 4.0, 2.0: 3.0, 3.0: 2.0, 4.0: 1.0, 5.0: 0.0,
            6.0: 0.0, 7.0: 0.0, 8.0: 0.0, 9.0: 0.0, 10.0: 0.0, 11.0: 0.0,
            12.0: 0.0, 13.0: 0.0, 14.0: 0.0, 15.0: 0.0, 16.0: 0.0, 17.0: 0.0,
            18.0: 0.0, 19.0: 0.0,
        }
        rankrel_map = {
            0.0: 3.0, 1.0: 2.0, 2.0: 1.0, 3.0: 0.0, 4.0: 0.0, 5.0: 0.0,
            6.0: 0.0, 7.0: 0.0, 8.0: 0.0, 9.0: 0.0, 10.0: 0.0, 11.0: 0.0,
            12.0: 0.0, 13.0: 0.0, 14.0: 0.0, 15.0: 0.0, 16.0: 0.0, 17.0: 0.0,
            18.0: 0.0, 19.0: 0.0,
        }
        rankrel_map = {
            0.0: 1.0, 1.0: 0.0, 2.0: 0.0, 3.0: 0.0, 4.0: 0.0, 5.0: 0.0,
            6.0: 0.0, 7.0: 0.0, 8.0: 0.0, 9.0: 0.0, 10.0: 0.0, 11.0: 0.0,
            12.0: 0.0, 13.0: 0.0, 14.0: 0.0, 15.0: 0.0, 16.0: 0.0, 17.0: 0.0,
            18.0: 0.0, 19.0: 0.0,
        }
        scores = {}
        for col in [f'FM{k}'for k in range(2, 11)]:
            this_scores = []
            for _, rpred in tqdm(self.data.groupby(GROUP_KEY)):
                # ranking = np.argsort(rpred.loc[:, 'obs'].values) + 1.0
                # rels = np.array([rankrel_map[rank] for rank in ranking])
                rels_pred = pd.Series(np.argsort(rpred.loc[:, col].values)).map(rankrel_map)
                # print(np.sum(np.isnan(rels_pred.values)))
                score = nDCG(rels_pred, topk=1)
                # print(f'Type of score: {type(score)}')
                this_scores.append(float(score))
            # pprint(this_scores)
            # pprint(np.nanmean(this_scores))
            scores.update({col: np.nanmean(this_scores)})
            # scores[col] = js
        pprint(scores)
        pass

    def test_rmse(self):
        y_obs = self.data.loc[:, 'obs'].values
        rmse_scores = {}
        best_model = None
        best_score = None
        for col in [f'FM{k}' for k in range(2, 11)]:
            y_pred = self.data.loc[:, col].values
            score = mean_squared_error(y_obs, y_pred)
            if (best_score is None) or (score < best_score):
                best_score = score
                best_model = col
            rmse_scores.update({col: score})

        for col in [f'CDFM{k}' for k in range(2, 5)]:
            y_pred = self.data.loc[:, col].values
            score = mean_squared_error(y_obs, y_pred)
            if (best_score is None) or (score < best_score):
                best_score = score
                best_model = col
            rmse_scores.update({col: score})

        print('RMSE Evaluation')
        pprint(rmse_scores)
        print(f'+ Best Model: {best_model}')
        print(f'+ Best Score: {np.round(best_score, 5)}')



if __name__ == '__main__':
    unittest.main()
