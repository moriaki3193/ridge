# ridge
ランキング学習と推薦アルゴリズムに特化したPython機械学習ライブラリです．
パッケージ名は，作者(@moriaki3193)が大好きなレースゲーム`R4`に由来します．

## 必要なもの
- Python 3.6.0 ~
- NumPy
- Scipy

## 実装済みモデル
`ridge`では次の機械学習モデルが実装されたクラスを利用可能です．

| Class | Module | Description | Document |
|:------|:-------|:------------|:---------|
| FMRegressor | factorization_machines | 回帰タスクを学習するFactorization Machine | [FMs](./docs/FactorizationMachines.md) |
| FMClassifier | factorization_machines | 分類タスクを学習するFactorization Machine | [FMs](./docs/FactorizationMachines.md) |
| MatFac | matrix_factorization | 行列分解 | [MFs](./docs/MatrixFactorization.md) |
| NNMatFac | matrix_factorization | 非負値行列分解 | [MFs](./docs/MatrixFactorization.md) |
| ConditionalLogit | logit_models | 条件付きロジットモデル．選択肢の間で特徴量が異なる離散選択モデルの一つ | [LMs](./docs/LogitModels.md) |