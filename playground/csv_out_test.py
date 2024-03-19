"""_summary_
学習したモデルをcsvに出力できるようにするため、配列でcsvに出力読み込みができるかテストを行うファイル

numpy配列は多次元配列でcsvに出力できないから、それぞれのパラメータごとに１次元化してcsvに収める作戦

C#とかで扱うなら、jsonで多次元配列のままがいいかもー＞JSONでやった方が良さそう
"""

import numpy as np
import os, sys
sys.path.append(os.pardir)
from two_layer_net_backp import TwoLayerNet

if __name__ == "__main__":
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    test_w1_flatten = np.ravel(network.params['W1'])
    test_w2_flatten = np.ravel(network.params['W2'])
    # test_b1_flatten = np.ravel(network.params['b1']) # バイアスはそもそも1次元なのでflattenしなくていい
    print(test_w1_flatten)
    print(test_w2_flatten)
    print(test_w1_flatten.shape, test_w2_flatten.shape)

    test_w1_reshape = test_w1_flatten.reshape(784, 50)
    test_w2_reshape = test_w2_flatten.reshape(50, 10)
    print(test_w1_reshape.shape, network.params['W1'].shape)
    print(test_w2_reshape.shape, network.params['W2'].shape)
