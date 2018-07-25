# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats

# stats.norm.pdf(x, loc=0, scale=1)


class SoftRank:
    """SoftRank: Optimizing Non-Smooth Rank Metrics (2008).

    モデルの出力は，エンティティのリスト(とそれを生成する確率)．
    その際の入力は，エンティティの素性のリスト(行列？)．
    """

    def __init__(self):
        pass

    def predict(self, X):
        """Prediction with a given features.

        Parameters
        ----------
        X : np.ndarray, whose shape is (n_entities, n_features)
        """
