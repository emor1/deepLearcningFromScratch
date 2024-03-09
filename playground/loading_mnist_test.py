import os, sys
sys.path.append(os.pardir)

from dataset.mnist import load_mnist


# 最初の呼び出し
# (訓練画像、訓練ラベル), (テスト画像、テストラベル)
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

# load_mnist(normalizeにより０〜１に正規化、Falseにより255のまま、flattenにより1次元配列に)