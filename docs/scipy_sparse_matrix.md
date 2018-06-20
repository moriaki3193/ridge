# Scipy Sparse Matrix のメモ
## 目的
疎な特徴量を，普段の密な形式で表現する場合には，
大量の`0`が値を占めることになり，メモリを大幅に食い尽くす．
たった10MB程度のログデータを，ユーザやアイテムのone-hot形式に
落としこもうとすれば最後，手元のマシンには乗り切らないほどの
大きなデータとなってメモリを占領することになる．

そのような問題を解決するには，スパースなデータを効率的に管理・操作する
データ構造を利用するのが一番である．
今回はその一つとして，`scipy.sparse`の様々な疎行列と基本的な操作方法
についてまとめたい．

## scipy.sparse.lil_matrix
行方向にリストが連結された疎行列の形式．`lil`は`List of lists`の省略系．
`lil_matrix(arg1, shape=None, dtype=None, copy=False)`のように初期化する．
いくつか具体例を取り上げる．

```Python
from scipy.sparse import lil_matrix

d = lil_matrix(D)  # dense matrix から初期化する
s = lil_matrix(S)  # 他の形式の疎行列のインスタンスから生成する．S.tolil()と同じ．
mn = lil_matrix((M, N), [dtype])  # M行N列の空の行列を生成する．
```

### Note
疎行列では，`addition` `subtraction` `multiplication` `division` `power` といった
算術演算をサポートしている．

#### LIL形式の利点
- 柔軟なスライシングに対応．
- 疎行列に対する構造の更新が効率的．

#### LIL形式の欠点
- `LIL` + `LIL` の演算が低速．
  - `CSR`か`CSC`形式の行列を利用することを考える．
- 列方向のスライシングが低速．
  - `CSR`を利用せよ．
- LIL行列とベクトルの積が低速．
  - `CSR`か`CSC`形式の行列を利用することを考える．

#### 想定された利用方法
- 疎行列を生成するのに適している．
- 生成したら，算術演算が高速な`CSR`や`CSC`への変換が可能．
- 巨大な疎行列を生成する場合には，`COO`形式の利用を考えて．

### 使用例
```Python
from scipy.sparse import lil_matrix

lol = [[1, 0, 1, 0],
       [0, 0, 0, 1],
       [0, 0, 1, 0]]
lil = lil_matrix(lol)

print(lil.data)
# → array([list([1, 1]), list([1]), list([1])], dtype=object)

print(lil.rows)
# → array([list([0, 2]), list([3]), list([2])], dtype=object)

csr = lil.tocsr()
csc = lil.tocsv()
```

## scipy.sparse.coo_matrix
値と座標を指定して作成する形式の疎行列．
`ijv`あるいは`triplet`形式とも呼ばれる．
インスタンスの生成にはいくつかの方法がある．

```Python
from scipy.sparse import coo_matrix

d = coo_matrix(D)  # dense matrixを引数にとるケース
s = coo_matrix(S)  # 他の疎行列のインスタンスから生成するケース
mn = coo_matrix((M, N))  # M行N列の空の行列を生成するケース
dat = coo_matrix((data, (i, j)))  # 値と座標を与えるケース
```

### Note
`LIL`形式と同様に，算術演算をサポートしている．

#### COO形式の利点
- 疎行列の変換を素早く操作可能．
- 重複する要素のエントリーを許す．
- `CSR`ないし`CSC`形式の疎行列からの変換，またそれらの形式への変換が高速．

#### COO形式の欠点
- 算術演算とスライシングを直接サポートしていない．

#### 想定された利用方法
- 疎行列を組み立てる際に，非常に高速な形式である．
- 高速な算術演算や，線形代数操作を実現する．
- `CSR`や`CSC`形式に変換する際に，重複した`(i, j)`要素をまとめてくれる．

### 使用例
```Python
# 空の行列を組み立てる
coo_matrix((3, 4), dtype=np.int8).toarray()

# triplet形式を利用したコンストラクト
row = np.array([0, 3, 1, 0])
col = np.array([0, 3, 1, 2])
dat = np.array([4, 5, 7, 9])
coo_matrix((data, (row, col)), shape=(4, 4)).toarray()

# 同一の座標に値が重複しているケースの挙動
row  = np.array([0, 0, 1, 3, 1, 0, 0])
col  = np.array([0, 2, 1, 3, 1, 0, 0])
data = np.array([1, 1, 1, 1, 1, 1, 1])
coo = coo_matrix((data, (row, col)), shape=(4, 4))
coo.toarray()
# (0, 0)成分の値が集計されていることがわかる
# array([[3, 0, 1, 0],
#        [0, 2, 0, 0],
#        [0, 0, 0, 0],
#        [0, 0, 0, 1]])
```

## scipy.sparse.csr_matrix
行方向に圧縮された疎行列の形式．

```Python
from scipy.sparse import csr_matrix

s = csr_matrix(S)  # 他の形式の疎行列を利用した生成

# 標準的なCSR形式の表現方法
# `i'行に含まれる列方向のインデックスは
# `indices[indptr[i]:indptr[i+1]]
# に格納され，それらに対応する値は
# data[indptr[i]:indptr[i+1]]
# に格納される．
# shapeパラメータが与えられていない場合は，
# インデックス配列から推定される．
indptr = np.array([0, 2, 3, 6])
indices = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
csr_matrix((data, indices, indptr), shape=(3, 3)).todense()
# matrix([[1, 0, 2],
#         [0, 0, 3],
#         [4, 5, 6]])
```

### Note
種々の算術演算をサポートしている．

#### CSR形式の利点
- `CSR` + `CSR`や`CSR` * `CSR`などの演算が高速．
- 行方向のスライシングが高速．
- ベクトルとの内積演算が高速．

## その他のTips
### 疎行列のままdump & read
```Python
np.save('sparse_matrix', sparse_matrix)  # ← 独自のフォーマットで保存される．
loaded_sparse = np.load('sparse_matrix.npy')
```

### 疎行列を密行列に変換する
```Python
dense = sparse.todense()
```

### 非0の要素数を取得する
```Python
print(sparse_mat.nnz)  # → 非0の要素数を出力
```
