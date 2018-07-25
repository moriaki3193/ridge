<!-- 
#D8DBE2
#A9B5D1
#5772AF
#373F51
#1B1C1E
-->

# Hodwild!
このメモは[1]の翻訳記事に近いものである．
Field-aware Factorization Machineの実装の一つである[2]では，モデルの学習プロセスを高速化する為にHogwild!の手法を利用している．
しばしばモデルの学習には長い時間を費やし，そのために試行錯誤のプロセスに割く時間が削られてしまう．
そのような経験を緩和する為に，一つのアプローチとしてこの手法に対する理解を深め，`ridge`をより良いものにすることを目指す．

## Overview
Hogwild!は非同期的な確率的勾配降下法のアルゴリズムの一つである．
この手法では，`lock-free`(訳注: ある勾配を計算する場合に，その他の勾配の計算を妨げることをしないよう)な勾配の更新アプローチをとる．
機械学習のモデルの観点からすれば，重みは複数のプロセスから同時に**互いに上書きされる可能性を伴って**更新されうることを意味している．
この記事では，Hogwild!をPythonで実装するために，`multiprocessing`[3]ライブラリを利用し，線形回帰モデルを学習させる方法について紹介したいと思う．

あまりに難しく込み入ったことを紹介してこの記事の読者を混乱させてしまうことがないように，読者はすでに勾配降下法のコンセプトと，非同期的プログラミングがどのようなものかをある程度把握しているものとする．

## Hogwild! Explained
勾配降下法は反復的なアルゴリズムである．処理を高速化するためには，ひょっとしたらGPUなどのより高速な処理系にその反復的な処理を投げる必要があるかもしれない．
Hogwild!は確率的勾配降下法の手法をより大きなデータセットを利用した学習に対しても有効にするための，並行的な手法だ．
その基本的なアルゴリズムは一見してシンプルなものである．
前提とする条件は，ニューラルネットワークであれ，線形回帰であれ，マシンに搭載されるプロセッサが重みを同時に読み込めて，なおかつ更新できるということである．
端的に言えば，重みがこれ以上分割できない構成要素の加算が可能な形式で共有メモリに格納されていることが必要ということだ．
この手法が提案された論文から引用すれば，個々のプロセッサの確率的勾配降下法による重みの更新は，次のような擬似コードで表現される．

```
loop
    トレーニング集合から一つの事例 (X_i, y_i) を取り出す
    (X_i, y_i) における勾配を評価する
    要素ごとに重みの更新を行う
end loop
```

それでは，このコードを複数のプロセッサで走らせ，同じモデルの重みを同時に更新する．
ランダムに動き回り，互いのつま先を踏みつけあっている数多くの農場の動物を監視しているような挙動から，
このアルゴリズムがHogwild!と呼ばれる所以がわかるだろう．

このアルゴリズムが実現するための条件について今回我々は特に重きを置かないが，いくつかあるうち大抵重要なものは，
勾配の更新はスパースであり，勾配の更新が行われる0ではない要素の数が少数でなければならないことだ．
これは学習に放り込むデータがスパースである場合によく当てはまるだろう．
更新の頻度が疎であるということは，異なるプロセッサが互いのつま先を踏みつけ合うことが少ないということである．
換言すれば，同じ重みを更新するために，同時にその重みにアクセスすることが少ないということだ．
ただし，その可能性が比較的小さいというだけで，衝突(訳注: 同じ重みに同時にアクセスすること)は起こる．
しかしながら，実際にはそれが正則化の役割を担うものとして解釈できるのである．

エクササイズに当たって，線形回帰モデルをHogwild!を利用した学習を実装していく．
[GitHubのリポジトリ](https://github.com/srome/sklearn-hogwild)には，scikit-learnを有効活用した実装が全て掲載されている．
この投稿においては，リポジトリのコードの一部を抜粋するので，適せんリポジトリの内容を参照してほしい．

## Linear Regression Refresher
私と読者の間で共通の認識ができるように，ここでは以降の説明で利用する記法について説明したい．
線形回帰モデル*f*は，*n*次元の実数ベクトル`x`を入力として受け取り，実数`y`を以下のような式を通じて出力する．

<p class='latex'>
    f(<b>x</b>) = <b>w</b>・<b>x</b>
</p>

上の式における`w`が、このモデルにおいて学習させたい重みのベクトルである．
損失関数としては，2乗誤差を利用し，データセットの一つの事例に対する損失は次のように書き表すことができる．

<p class='latex'>
    <i>l</i>(<b>x</b>, y) = {f(<b>x</b>) - y}<sup>2</sup>
</p>

確率的勾配降下法を通じてモデルを学習させるために，以下のように勾配更新ステップを計算する必要がある．

<p class='latex'>
    <b>w</b><sub>t+1</sub> = <b>w</b><sub>t</sub> - λG<sub><b>w</b></sub>(<b>x</b>, y)
</p>

ここでいうλとは学習率であり，G<sub><b>w</b></sub>は損失<i>l</i>の<b>w</b>についての勾配の期待値である．すなわち

<p class='latex'>
    E[G<sub><b>w</b></sub>(<b>x</b>, y)] = ∇<sub><b>w</b></sub>(<b>x</b>, y)
</p>

特に，損失が2乗誤差である場合には，次のように計算できる．

<p class='latex'>
    G<sub><b>w</b></sub>(<b>x</b>, y) = -2(<b>w</b>・<b>x</b> - y)<b>x</b> ∈ R<sup>n</sup>
</p>

## Generating our Training Data
まずはトレーニングデータを手早く作ってしまおう．
今回は，データは我々が用意するモデルと同様(つまり線形回帰モデル)の関数から生成されると想定する．
このモデルの重みを，<b><i>w</i></b>と呼ぶことにする．

```Python
import scipy.sparse

n = 10  # The number of features
m = 20000  # The number of training examples

X = scipy.sparse.random(m, n, density=.2).toarray()  # 特徴量の次元の2割しかnon-zeroの値が入っていないもの
real_W = np.random.uniform(0, 1, size=(n, 1))  # 本当の重みベクトルを定義する
X = X / X.max()
y = np.dot(X, real_W)
```

### High Level Approach
`multiprocessing`ライブラリを利用することで，コードを並行的に処理するための追加的な処理が必要になる．
`Pool`の`map`関数は，勾配を計算し，更新することを非同期的に行うことを可能にするためのものだ．
アルゴリズムの鍵となる要素は，その重みベクトルは全てのプロセスから同時に，ロックされることがなくアクセス可能であるということだ．

### Lock-free Shared Memory in Python
最初に，共有メモリ中に重みベクトル<b>w</b>を定義し，アクセスをロックすることなく実現する必要がある．
この機能を実現するために，`miltiprocessing`ライブラリから，`sharedctypes.Array`クラスを利用しなければならないだろう．
加えて，`numpy`の`frombuffer`関数を利用して，`numpy`の配列からもアクセスができるようにする．

```Python
from multiprocessing.sharedctypes import Array
from ctypes improt c_double
import numpy as np

coef_shared = Array(c_double,
        (np.random.normal(size=(n, 1)) * 1. / np.sqrt(n)).flat,
        lock=False)  # Hogwild!
w = np.frombuffer(coef_shared)
w = w.reshape((n, 1))
```

最終的な並行的コードに向けて，`multiprocessing.Pool`を利用して勾配の更新を複数のワーカが受け持てるように割り当てる．
このベクトルを共有メモリにさらけ出すことで，ワーカは仕事をすることが可能になるのだ．

### Gradient Update
私たちの最終目標は，並行的に勾配を更新することを実現することだ．
そのためには，更新とは何かを定義してやる必要がある．
この[リポジトリ](https://github.com/srome/sklearn-hogwild)では，
複数のプロセスに重みベクトルを公開するために，もっと込み入ったアプローチを取っているが，複雑な技術の説明を避けるために`multiprocessing.Pool.map`の挙動を確かめやすいグローバル変数を利用する．
その`map`関数の名前空間をグローバルなものにしてくれる．
同一の重みベクトル<b>w</b>を全てのワーカで共有するには，これで十分機能する．

```Python
learning_rate = .001
def mse_gradient_step(X_y_tuple):
    global w  # 説明のためにグローバル変数を利用している．
    X, y = X_y_tuple

    # 勾配の計算
    err = y.reshape((len(y), 1)) - np.dot(X, w)
    grad = -2 * np.dot(np.transpose(X), err) / X.shape[0]

    # 0ではない要素について一度に更新をかける
    for index in np.where(abs(grad) > .01)[0]:
        coef_shared[index] -= learning_rate * grad[index, 0]
```

### Preparing the examples for multiprocessing.Pool
トレーニング事例をタプルに切り出し，事例ひとつごとにワーカに引き渡す必要がある．

```Python
batch_size = 1
examples = [None] * int(X_shape[0] / float(batch_size))
for k in range(int(X.shape[0] / float(batch_size))):
    Xx = X[k * batch_size : (k + 1) * batch_size, :].reshape((batch_size, X.shape[1]))
    yy = y[k * batch_size : (k + 1) * batch_size].reshape((batch_size, 1))
    examples[k] = (Xx, yy)
```

### The Asyncronous Bit
最終的なHogwild!による学習のコードは次のようになる．

```Python
from multiprocessing import Pool

# Hogwild!を利用した学習
p = Pool(5)
p.map(mse_gradient_step, examples)

print(f'Loss function on the training set: {np.mean(abs(y - np.dot(X, w)))}')
print(f'Difference from the real weight vector: {abs(real_w - w).sum()}')
```

結果は次のようになる．

```Text
Loss function on the training set: 0.0014203406038
Difference from the real weight vector: 0.023417324317
```

## Importance of Asynchronous Methods
あなた方もご存知の通り，深層学習が大変な人気を博している．
深層学習を実現するためのニューラルネットワークでは，確率的勾配降下法を利用して学習を行なっている．
深層学習には大規模なデータセット(に加えて強化学習では対話的な環境)を必要とする．
規模が大きくなればなるほど，学習により長い時間がかかり，厄介なものとなってしまう．
したがって，ニューラルネットワークを並行して学習させることは大きな課題なのだ．

確率的勾配降下法の並行化にはHogwild!とは異なったアプローチも存在する．
Googleは並行的に確率的勾配降下法を実現するために[Downpour SGD](https://static.googleusercontent.com/media/research.google.com/en//archive/large_deep_networks_nips2012.pdf)を開発した．


特筆すべきは，非同期的な手法は発展途上であり，今後数年間にどのような新しい進歩が起こるかわからないということである．
現在のところ，非同期的な手法で見られる精度の向上に関する数学的証明はあまり存在しない．
実際，多くのアプローチは理論的に正当化されていないが，その効果は印象的なアプリケーションの実装により示されている．
数学者にとっては，これらの方法を形式的な説明に挑戦する時に，新しい数学が生まれうるかということは興味深いことである．

## Reference
1. [Hogwild!? Implementing Async SGD in Python](https://srome.github.io/Async-SGD-in-Python-Implementing-Hogwild!/)
2. [libFFM - A Library for Field-aware Factorization Machines](https://github.com/guestwalk/libffm)
3. [mulitprocessing - Python](https://docs.python.jp/3/library/multiprocessing.html)