# Cython
アルゴリズムだけで高速化できない，言語の壁を超えたパフォーマンスの実現のためにCythonを学ぶ．

## 専門用語
| 用語 | 説明 |
|:---:|:-----|
| 拡張モジュール | Pythonで利用できるコンパイル済みのモジュールのこと |
| .pyx | 実装ファイルには拡張子 .pyx をつける |
| .pyx | 定義ファイルには拡張子 .pxd をつける |
| .pyi | includeファイルには拡張子 .pxi をつける |


## Cythonのコンパイル
PythonインタープリタはCythonを直接インポートし，実行することができない．
Pythonで利用可能なコードをCythonから作成するには，コンパイルパイプラインと呼ばれる一連の処理を行う必要がある．
コンパイルパイプラインは，CythonのコードをPythonのインタープリタが使える`Python拡張モジュール`に変換してくれる．

Cythonのコードを書いてから，Pythonで利用できるようになるまでの手順は，次のようにまとめられるだろう．

1. Cythonのコードを書く
2. コンパイルパイプラインを通じてCythonからC，C++のコードに変換し，それらをコンパイルする
3. コンパイルした結果を拡張モジュールとしてPythonで利用する

### cimport
`cimport`文は，`*.pyd`定義ファイルを参照し，その中の宣言をインポートする．
`import`文はPythonレベルで動作し，インポートは実行時に行われる．

`as`節を利用すれば，`cimport`された特定の宣言にエイリアスを与えることができる．

```Python
cimport simulator as sim

cdef sim.State st = sim.State(params)
cdef sim.real_t dt = 0.01
sim.step(st, dt)
```

## プロジェクトへの組み込み
`foo.pyx`という名前のCythonコードを，Pythonで利用可能なPythonコードに変換するケースを考える．

### 1. setup.pyを記述する
```Python
from setuptools import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize('foo.pyx'))
```

`cythonize`がやっていることは，`.pyx`ソースファイルを対象としてCythonコンパイラを呼び出すこと．
次に，`setup`関数が`ext_modules`として`cythonize`が吐き出すCまたはC++のコードをコンパイルし，拡張モジュールに変換する．

### 2. コンパイルする
コマンドラインで`setup.py`を呼び出す．
`--inplace`フラグを利用して，`.pyx`のソースファイルと同じディレクトリに，対応する拡張モジュールを書き出す．

```shell
$ python setup.py build_ext --inplace
```
