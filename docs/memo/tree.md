# Tree Algorithms
## Overview
![Graph](https://www.analyticsvidhya.com/wp-content/uploads/2016/04/dt.png)

* * *

- Tree-basedな学習アルゴリズムはターゲットと特徴量の間の非線形の関係性を学習可能
- カテゴリ変数と実数変数の両方を入力特徴量として利用できる
- 特徴量がターゲットに与える影響を説明しやすい
- ノンパラメトリックな手法

### Terminology
![Graph2](https://www.analyticsvidhya.com/wp-content/uploads/2015/01/Decision_Tree_2.png)

* * *

- Root Node : 標本全体を表す
- Splitting : ノードを2つ以上のサブノートに分割すること
- Decision Node : 自身の子となるサブノートを持つサブノートのこと
- Terminal Node : これ以上分割されないノードのこと

### 回帰木の分割
ターゲットが連続的な実数変数である場合のノード分割のアルゴリズムについて考える。

サンプルを分割する基準としては、分散が低い特徴量と閾値での分割が選ばれる。
1. 各ノードの分散を計算する
2. 各ノードの分散の加重平均として、「分割」の分散を計算する

実際には、「分割」の間でサンプル数は共通しているので、次のようにして損失の「代理」が求められる。

```Python
lt_samples = [...]
gte_samples = [...]

lt_mean = np.mean(lt_samples)
gte_mean = np.mean(gte_samples)

Sj = np.sum(np.square([elem - lt_mean for elem in lt_samples])) \
    + np.sum(np.square([elem - gte_mean for elem in gte_samples]))
```

### パラメータ
いずれも過学習を防ぐためのモデルのパラメータである。

- ノード分割の最小サンプル数
- ターミナルノードの最小サンプル数
- 木の最大の深さ
- ターミナルノードの最大数
- 分割で考慮する特徴次元の最大数 ← 高速化にも役立つ

### Pruning (剪定、枝刈り)
ノード分割時に前処理を挟むことで、その先の分割ステップが効率的になるようにする手法。
`scikit-learn`ではまだ実装されていないらしい。

1. 深さが大きく設定された決定木を作成する
2. 末端のノードから開始し、その親の分割で与えられる損失が自身よりも大きい場合には、親での分割を除去する

#### vs Linear Regression
1. 従属変数と独立変数の関係が線形モデルで近似できる場合には、線形モデルの方が良い
2. 従属変数と独立変数の間に非線形の関係がある場合には、回帰木の方が良い
3. なぜそのような出力が行われるのかについては、回帰木の方が解釈がしやすい

### GBMの擬似コード
```text
1. 結果を初期化する(ノードの分割をクリア)
2. for 1 to n_trees:
   2.1. 直前のターゲットに基づいて重みを更新する
   2.2. 選択されたサブサンプルについてのモデルを学習する
   2.3. 観測されたすべてのサンプル集合について予測を行う
   2.4. 学習率を考慮して、現在の結果に基づいて出力を更新する
3. 最終的な出力を返す
```

### 実装
- XGBoost
- scikit-learn

## References
- [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/#one)