"""CNNの学習
参考：https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/ch07/train_convnet.py
"""

import numpy as np
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt
from ch7_CNN import SimpleConvolutionNet

from tqdm import tqdm

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

max_epochs = 20

network = SimpleConvolutionNet(input_dim=(1,28,28),conv__param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1}, hidden_size=100, output_size=10, weight_init_std=0.01)

iters_num = 5000
batch_size = 100
learning_rate = 0.1
train_size = x_train.shape[0]

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_eppoch = max(train_size/batch_size, 1)

for i in tqdm(range(iters_num)):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    grad = network.gradient(x_batch, t_batch)

    # 更新
    for key in ('W1', 'b1', 'W2', 'b2','W3','b3'):
        network.params[key] -=learning_rate*grad[key]

    loss = network.loss(x_batch, t_batch)

network.save_params('cnn_param,json')
