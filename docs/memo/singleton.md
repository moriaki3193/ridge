# The Singleton
[ここ](http://python-3-patterns-idioms-test.readthedocs.io/en/latest/Singleton.html)の解説の翻訳．

## Simplest
もっともシンプルなデザインパターンといえば，シングルトンかもしれない．
シングルトンでは，特定の型のオブジェクトがたった一つしかインスタンス化されないという制約を実装する．
これを実現するために，プログラマの操作の手から，オブジェクトの生成を管理する必要が生じる．
一つ便利な方法は，クラスの定義の内部にネストしたプライベートなクラスのインスタンスをデリゲートすることだ．

```Python
# Singleton/SingletonPattern.py

class OnlyOne:
    
    class __OnlyOne:
        def __init__(self, arg):
            self.val = arg

        def __str__(self):
            return repr(self) + self.val

    instance = None

    def __init__(self, arg):
        if not OnlyOne.instance:  # インスタンスが一つもないケース
            OnlyOne.instance = OnlyOne.__OnlyOne(arg)
        else:  # インスタンスがすでに存在するケース
            OnlyOne.instance.val = arg

    def __getattr__(self, name):
        return getattr(self.instance name)


if __name__ == '__main__':
    x = OnlyOne('sausage')
    print(x)
    y = OnlyOne('eggs')
    print(y)
    z = OnlyOne('spam')
    print(z)
    print(x)
    print(y)
    print(`x`)
    print(`y`)
    print(`z`)

# <__main__.__OnlyOne instance at 0076B7AC>sausage
# <__main__.__OnlyOne instance at 0076B7AC>eggs
# <__main__.__OnlyOne instance at 0076B7AC>spam
# <__main__.__OnlyOne instance at 0076B7AC>spam
# <__main__.__OnlyOne instance at 0076B7AC>spam
# <__main__.OnlyOne instance at 0076C54C>
# <__main__.OnlyOne instance at 0076DAAC>
# <__main__.OnlyOne instance at 0076AA3C>
```
