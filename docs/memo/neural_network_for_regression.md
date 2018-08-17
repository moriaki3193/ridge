# Neural Network for Regression Tasks
[この記事](https://medium.com/@rajatgupta310198/getting-started-with-neural-network-for-regression-and-tensorflow-58ad3bd75223)を参考にNeural Networkを簡単におさらいする。

## Neural Networkとは
![feed forward neural network](https://cdn-images-1.medium.com/max/1600/1*LaEgAU-vdsR_pClMcgbikQ.jpeg)

- たいていの関数の近似に応用できる
- 隠れ層が一つ以上あるものをDeep Neural Netとよび、これが昨今巷を騒がせている深層学習モデル
- 各層は任意の個数のノードを持っている
    - 入力層はデータの特徴次元数と同じ個数のノードを持つ
    - 隠れ層は任意の個数のノードを持つ
    - 出力層は...
        - 回帰の場合は1つ
        - 分類の場合は1つ以上のノードを持つ
        - 分類の場合は実質的には出力したいクラスの個数がノードの数に一致することが多い
- 活性化関数としては...
    - sigmoid
    - tan hyperbolic
    - linear
    - ReLU

### Feed-forwardの意味
[Wikipedia](https://en.wikipedia.org/wiki/Feedforward_neural_network)によると...

> A feedforward neural network is an artificial neural network wherein connections between the nodes dot not form a cycle.

ニューロン同士がサイクルを形成している`recurrent`とは区別される。

### Backpropagationの意味
`誤差逆伝播法`。
損失関数から得られるコストを利用して、ノード間の重みを学習する手法。
