"""
学習したモデルを読み込み予測するテスト
"""

import numpy as np
from dataset.mnist import load_mnist
from two_layer_net_backp import TwoLayerNet

from PIL import Image

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10, load=True)

img_num = 19
print(t_test[img_num])
print("predict: ", np.argmax(network.predict([x_test[img_num]])))
# pil_img = Image.fromarray(np.uint(x_test[img_num].reshape(28,28)*255))
# pil_img.show()
