"""_summary_
学習したモデルをcsvに出力できるようにするため、配列でcsvに出力読み込みができるかテストを行うファイル

"""

import numpy as np
import os, sys
sys.path.append(os.pardir)
from two_layer_net_backp import TwoLayerNet

if __name__ == "__main__":
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    print(network.params['W1'])
