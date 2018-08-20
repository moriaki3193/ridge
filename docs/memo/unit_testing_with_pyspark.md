# Unit Testing with PySpark
[この記事](https://blog.cambridgespark.com/unit-testing-with-pyspark-fb31671b1ad8)のまとめ。

PySparkのジョブのユニットテストをいかに効率的に書くのかについてのノウハウをまとめたもの。
そのためには、コードをどのように構造化していくかが重要になる。

PySparkのコードを書く研究者はたいてい一つの巨大な関数を書きがち。

```Python
def my_application_main():
    #Code here to create my spark contexts
    
    #Code here to fetch initial tables/rdds
    
    #Data Science magic code...
    #...
    #...
    #...
    
    #Persist/send results
```

このような関数を相手にテストを書こうとすると、すぐにテストが不可能であることに気がつくはず。
なぜならば、コンテキストの作成がジョブに張り付いており、かつローカルの環境ではそれを作成できないからである(`sc` not found 問題?)。
コツは、プログラムを小さな複数のユニットに分割し、実際のSpark Contextに依存しないコードを書くことである: つまりローカルマシンで試すことができれば良いのだ。

```Python
def my_logic_step_1(my_rdd, my_dataframe):
    #some processing
    ;
    
def my_logic_step2(my_dataframe):
    #some processing
    ;
    
def persist_results(my_dataframe):
    #persist back results
    ;
    
def logic_main(ctx):
    #get tables, rdds in interest from the context
    data = my_logic_step_1(x, y)
    data = my_logic_step_2(data)
    persist_results(data)
    
def my_application_main():
    ctx = ...#create spark context as you see fit
    logic_main(ctx)
```

主な考え方としては、RDDやDataFrameを引数に取る小さな関数を積み重ねていくことである。
テストが簡単になり、これらモジュールを組み合わせることで実現したいロジックの構築が可能。
しかし、依然として`SparkContext`が入力されることを期待している。
これらモジュールを実際にロジックを実現する時のように組み合わせることで統合テストが可能である点も特徴的である。

テスト可能なユニット単位にジョブを分割し終えたら、テストのためにローカルのSparkContextを生成できる。
テストに関係のない警告を排除するために、`py4j`のログレベルを下げることをおすすめする。

```Python
import logging
from pyspark.sql import SparkSession


def suppress_py4j_logging():
    logger = logging.getLogger(‘py4j’)
    logger.setLevel(logging.WARN)

def create_testing_pyspark_session():
    return (SparkSession.builder
            .master(‘local[2]’)
            .appName(‘my-local-testing-pyspark-context’)
            .enableHiveSupport()
            .getOrCreate())
```

`unittest`をテストツールとして利用していることを前提にして、PySparkで簡単にテストの基本クラスを作成できることに注目してください。

```Python
import unittest
import logging
from pyspark.sql import SparkSession


class PySparkTest(unittest.TestCase):
 
    @classmethod
    def suppress_py4j_logging(cls):
    logger = logging.getLogger(‘py4j’)
    logger.setLevel(logging.WARN)

    @classmethod
    def create_testing_pyspark_session(cls):
        return (SparkSession.builder
                .master(‘local[2]’)
                .appName(‘my-local-testing-pyspark-context’)
                .enableHiveSupport()
                .getOrCreate())
 
    @classmethod
    def setUpClass(cls):
        cls.suppress_py4j_logging()
        cls.spark = cls.create_testing_pyspark_session()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()
```

この基本クラスを用意することによって、テストケースを次のように簡単に記述可能になる。

```Python
from operator import add


class SimpleTest(PySparkTest):

    def test_basic(self):
        test_rdd = self.spark.sparkContext.parallelize([‘cat dog mouse’,’cat cat dog’], 2)
        results = test_rdd.flatMap(lambda line: line.split()).map(lambda word: (word, 1)).reduceByKey(add).collect()
        expected_results = [(‘cat’, 3), (‘dog’, 2), (‘mouse’, 1)]
        self.assertEqual(set(results), set(expected_results))
```
