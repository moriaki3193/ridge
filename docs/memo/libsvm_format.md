# LIBSVM Data
[LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/)のデータセットのフォーマットについてまとめる。

LIBSVMフォーマットでは、分類、回帰、多値ラベルのデータを表現できる。
テキストベースのフォーマットで、1行が一つのサンプルに対応する。要素として0が入る特徴次元は明記されず、疎行列の表現に適している。
各行の最初の要素が予測のターゲットとなる値として解釈される。

## scikit-learn
scikit-learnでは[load_svmlight_file](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_svmlight_file.html)という実装があり、LIBSVMフォーマットのデータセットをCSR疎行列として読み込んでくれる。
テキストベースのソースをパースすることは大変コストがかかるので、同じデータについて何回も処理を施す場合には`joblib.Memory.cache`でloaderをラップすることが推奨される。

svmlightで`qid`として知られる、ペアワイズの情報を表現する場合では、`query_id`引数をTrueに設定する必要がある。

この実装はCythonで行われており、十分に高速だが、API互換のより高速な実装が[ここ](https://github.com/mblondel/svmlight-loader)にある。