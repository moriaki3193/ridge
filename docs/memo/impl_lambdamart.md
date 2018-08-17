# Implementation of LambdaMART
- 将来的にはPySparkを利用できるか否かで内部の挙動を変更できるようにする

## Questions
- LambdaMARTでのRegressionTreeへの入力はなにか？
- LambdaMARTでのRegressionTreeがFitするものはなにか？
    - そのFitをするときに、損失として設定されるものはなにか？
- LambdaMARTで、RegressionTreeをアンサンブルしていく時の、線形結合の係数はどのように決まる？
- LambdaMARTで、M個のRegressionTreeを生成した後に、出力はどのように行われる？

## Answers
- RankListの集合、つまりデータセット
- λの値、つまり`|MetricDiff_ij| × rho`で、rhoは`1 + exp(si - sj)`の **逆数**。
    - RegressionTreeが最小化するのは、λと
    - モデル全体が最小化しようとしている損失は交差エントロピー。
- p17アルゴリズムの記述の一番下の箇所。`learningRate`と属するノードの`gamma`の値の積となる。
- すべてのドキュメントについてFn(x)を計算し、並び変えた結果を出力する。

## jobandtalent RankLib
- `x` : 分散処理できない
- `o` : 分散処理可能
- `◇` : Sparkに実装がある(のでそれを利用する部分)

| Distributable? | step | sub step | description |
|:---------------|:-----|:---------|:------------|
| - | 1 | - | 入力: |
| - | 1 | a | ユーザ入力変数: `nTrees`, `nLeaves`, `learningRate` |
| - | 1 | b | ローカル変数: `featureHist`, `ensembleTrees` |
| - | 2 | - | for `m` in range(`nTrees`) |
| o | - | a | pseudo response を計算[computePseudoResponses](computePseudoResponses)する |
| o | - | b | 特徴量についてのヒストグラム(`featureHist`)を更新する　|
| ◇ | - | c | pseudo response, `featureHist` をもとに、RegressionTree `rt`を学習する |
| - | - | d | `ensembleTrees`に、RegressionTree `rt`を`learningRate`に基づいて追加[ensemble.add](ensemble.add)する |
| - | - | e | (Newton-Raphson法で計算されるγを使って)LambdaMARTとしての出力を更新[updateTreeOutput](updateTreeOutput)する |
| x | - | f | RegressionTree `rt`のサンプルをクリアする |
| x | - | g | 必要ならガーベジコレクションを動作させる |
| o | - | h | トレーニングデータについてモデルのスコアを計算(し表示)する |
| - | - | i | if `m` - bestModelOnValidationIdx > nRoundToStopEarly then 学習を終える |
| - | 3 | - | while `ensembleTrees`.`treeCount` > `bestModelOnValidation + 1` |
| - | - | a | `ensembleTrees`の末尾に存在するRegressionTreeインスタンスをPop |
| - | 4 | - | 出力: |
| o | 4 | a | 学習済みのモデル |

## computePseudoResponses
スレッドの数に応じて並行処理を挟むかどうかの判断をしている。

```Python
# self.pseudoResponses: List[float] ← λの集まり
# self.weights: List[float]

def computePseudoResponses(startIdx: int, endIdx: int, current: int):
    self.pseudoResponses = [0.0 for _ in range()]  # 長さは？
    self.weights = [0.0 for _ in range()]  # 長さは？

    cutoff: int = self.scorer.getK()  # nDCG@kとか、ERR@kのkを取得するということ
    for i in range(startIdx, endIdx + 1):
        # ?? orig → rl と変換を噛ませている理由 ??
        orig: RankList = samples.get(i)  # サンプルの単位は一つのクエリ結果
        indices: List[int] = MergeSorter.sort(modelScores, current, current + len(orig) - 1, False)  # スコアで並び替えた時のドキュメントのindexをリストで持っている
        rl: RankList = RankList(orig, indices, current)
        changes: List[List[float]] = scorer.swapChange(rl)
        for j in range(len(rl)):
            # NOTE j, kはモデルのスコアによって並び替えられたインデックスなので
            #      対応付けをし直す必要がある
            p1 DataPoint = rl[j]  # ランキングのj番目のドキュメント
            mj: int = indices[j]  # 6行上のやつ
            for k in range(len(rl)):
                if (j > cutoff) and (k > cutoff):
                    break
                p2: DataPoint = rl[k]  # ランキングのk番目のドキュメント
                mk: int = indices[k]
                if p1.getLabel() > p2.getLabel():
                    deltaNDCG: float = abs(changes[j][k])  # ← ERRとかの実装もできるように
                    if deltaNDCG > 0:
                        rho: float = 1.0 / (1 + np.exp(modelScores[mj] - modelScores[mk]))  # 論文p16下から2番目の式 lambdaの計算に利用する
                        lambda_ = rho * deltaNDCG  # p16下から3番目 損失をスコアで微分した結果
                        self.pseudoResponses[mj] += lambda_
                        self.pseudoResponses[mk] -= lambda_
                        delta: float = rho * (1 - rho) * deltaNDCG  # p16の1番下の式, この実装ではdeltaと呼んでいる
                        self.weights[mj] += delta
                        self.weights[mk] += delta
        current += len(orig)  # multi-threading の処理
```

## FeatureHistgram

## RegressionTree
`new RegressionTree(nLeaves, martSamples, pseudoResponses, hist, minLeafSupport)`.`fit()`している。
実装の初期化の部分を見ると、`RegressionTree(int nLeaves, DataPoint[] trainingSamples, double[] labels, FeatureHistgram hist, int minLeafSupport)`となっているので、**一つの回帰木の学習のターゲットは`lambda`**といえる。

`martSamples`はMARTの学習で利用する事例の集合ということ。

## updateTreeOutput
このメソッドの役割は今ある回帰木(λについてpoint-wiseに学習)の出力結果を更新すること。
Pythonでの実装なら、`gammas: List[float]`とかに格納していくのが良さそう。

```Python
def updateTreeOutput(rt: RegressionTree):
    leaves: List[Split] = rt.leaves()  # Splitは一つのノードの抽象クラス？
    for i in range(len(leaves)):
        s1: float = 0.0  # λの合計値を計算するための変数
        s2: float = 0.0  # wの
        s: Split = leaves[i]
        indices: List[int] = s.getSamples()  # そのノードに属するサンプルを集める
        for j in indices:
            s1 += self.pseudoResponse[j]
            s2 += self.weights[j]
        if s2 == 0:
            gammas[i] = 0.0
        else:
            gammas[i] = s1 / s2
```

## Emsemble.add
`ensemble.add(rt, learningRate)`

単にRegressionTreeと学習率を追加していっているだけ。