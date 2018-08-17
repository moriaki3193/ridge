# -*- coding: utf-8 -*-
# TODO DataPointクラスにlabelのプロパティを実装する

# BUG fitに与えられたトレーニングはRegressionTreeの内部でscikit-learnの
# DecisionTreeRegressorを利用する場合には、必ずdenseな形式に変換する
# Sparkが利用できず、DataFrameベースに学習を進められない場合には仕方がない？


IS_SPARK_AVAIABLE = True


class RegressionTree:  # TODO これはPySparkを上書きするでもOK

    @property
    def members(self):  # TODO return type annot
        """リーフノード番号そのノードに属するサンプルのインデックスを返す
        """
        pass


class Ranking:
    pass


class Scorer:
    """Scorer of Ranking.

    Attributes:
        k: 評価指標の@kの値
    """

    def __init__(self, k: int):
        self._k = k

    def swap_change(self, rl: Ranking):
        pass

    @property
    def k(self):
        return self._k


class LambdaMART:
    """LambdaMART

    Attributes:
        scorer: Rankingを評価指標に基づいてスコアリングする
        n_trees: アンサンブルするRegressionTreeの最大数
        n_leaves: 各RegressionTreeのリーフノードの数
        n_threshold: ノード分割時に保証するサンプル数の最小値
        learning_rate: Boosting Algorithmの学習率

    Methods:
        load: 学習済みのモデルを読み込む
        fit: モデルを学習する
        predict: ランキングを予測する

    References:
        BibTeX key: burges2010ranknet
    """

    def __init__(
            self,
            samples: RankingList,  # TODO 実装
            scorer: Scorer,
            n_trees: int = 1000,
            n_leaves: int = 10,
            n_threshold: int = 256,
            learning_rate: float = 1e-2):
        """pass
        """
        self.samples = samples
        self.scorer = scorer
        self.n_trees = n_trees
        self.n_leaves = n_leaves
        self.n_threshold = n_threshold
        self.learning_rate = learning_rate
        self.model_scores: List[float] = None
        # TODO クラス実装
        self._feature_hist = None
        self._ensemble_trees = None
        # TODO 長さの取得
        self._pseudo_responses = [0.0 for _ in range([])]
        self._weights = [0.0 for _ in range([])]
        #
        self._gammas = [0.0 for _ in range([])]

    @classmethod
    def _sort(
            scores: List[float],
            start_idx: int,
            end_idx: int,
            desc: bool = True) -> List[ind]:
        """スコアを並び替え元のインデックスのリストを返す

        Attributes:
            scores:
            start_idx:
            end_idx:
            desc:
        """
        pass

    @classmethod
    def _compute_rho(s1: float, s2: float) -> float:
        return 1.0 / (1.0 + np.exp(s1 - s2))

    def load(self, path2model: str):
        """学習済みのモデルを読み込む
        """
        # TODO impl
        pass

    def fit(self, verbose: bool = True):
        """モデルを学習する

        Parameters:
            verbose: 学習のログを表示するかどうか
        """
        self._calculate_pseudo_responses()
        self._feature_hist.update(self._pseudo_responses)  # TODO Histgramにupdateを実装する
        rt: RegressionTree = RegressionTree()  # TODO RegressionTreeのPySparkとscikit-learnでの実装
        # TODO 確認: ↑histはRankLibでの実装で利用されているだけで、SparkのDecisionTreeでは必要ない？
        self._ensemble_trees.add(rt, self.learning_rate)  # TODO Ensembleクラスにaddを実装する
        self._update_tree_output()

        rt.clear()  # TODO RegressionTree.clear()の実装

        # TODO バリデーションスコアを導出し、必要ならログを表示する

    def predict(self):
        """pass
        """
        # TODO impl
        pass

    def _calculate_pseudo_responses(self):
        """:math:`\lambda`を計算する

        RegressionTreeの学習時のターゲットとなる:math:`\lambda`と
        RegressionTree同士を線形結合する時の:math:`\gamma`の値の計算に
        必要なwの値を計算する。

        _compute_pseudo_responsesの計算をMulit-threadingに行う
        """
        self.pseudo_responses = [0.0 for _ in range([])]  # TODO
        self.weights = [0.0 for _ in range([])]  # TODO
        # TODO multi-threadingの導入
        self._compute_pseudo_responses(0, len(self.samples) - 1, 0)

    def _update_tree_output(self, rt: RegressionTree):
        # TODO PySparkのDecisionTreeModelには
        # 学習データが結局どのノードに属するかのメソッドが提供されていない
        # RegressionTreeの実装として、リーフノードに属するサンプルを集められるようにする

        # (ノード番号、そのサンプル)を返す機能を学習済みモデルに対するgetterとして実装する
        pass

    def _compute_pseudo_responses(
            self,
            start_idx: int,
            end_idx: int,
            current_idx: int):
        """指定されたRankingのリストについて\lambdaを計算する
        """
        cutoff: int = self.scorer.k
        for i in range(start_idx, end_idx + 1):
            orig: Ranking = self.samples[0]
            indices: List[int] = self._sort(
                    self.model_scores,
                    current,
                    current + len(orig) - 1)
            rl: Ranking = Ranking(orig, indices, current)  # TODO impl Ranking.__init__()
            delta_matrices: List[List[float]] = self.scorer.swap_change(rl)
            for j in range(len(rl)):
                lp1: DataPoint = rl[j]  # TODO DataPointはPySparkのLabeledPointに置き換える
                mj: int = indices[j]
                for k in range(len(rl)):
                    if (j > cutoff) and (k > cutoff):
                        break
                    lp2: DataPoint = rl[k]  # TODO 同上
                    mk: int = indices[k]
                    if lp1.label > lp2.label:  # TODO PySpark LabeledPointではlabel属性が実装されている
                        delta_metric: float = abs(delta_metrices[j][k])
                        if delta_metric > 0:
                            s1: float = self.model_scores[mj]
                            s2: float = self.model_scores[mk]
                            # p16 the 2nd Eq from the bottom
                            rho: float = self._compute_rho(s1, s2)
                            # p16 the 3rd Eq from the bottom
                            lambda_: float = rho * delta_metric
                            # p16 the 1st Eq from the bottom
                            # here, denoted as delta_
                            delta_: float = rho * (1.0 - rho) * delta_metric
                            # update
                            self._pseudo_responses[mj] += lambda_
                            self._pseudo_responses[mk] -= lambda_
                            self._weights[mj] += delta_
                            self._weights[mk] += delta_
            # update current index
            current += len(orig)
