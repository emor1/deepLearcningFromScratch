import cupy as cp

from functions_cupy import *
from gradient_cupy import *

class TwoLayerNetGPU:
    def __init__(self, icput_size, hidden_size, output_size, weight_initA_std = 0.01):
        # init params
        self.params = {}

        self.params['W1'] = weight_initA_std * cp.random.randn(icput_size, hidden_size)
        self.params['b1'] = cp.zeros(hidden_size)
        self.params['W2'] = weight_initA_std * cp.random.randn(hidden_size, output_size)
        self.params['b2'] = cp.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = cp.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = cp.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = cp.argmax(y, axis=1)
        t = cp.argmax(t, axis=1)

        accuracy = cp.sum(y==t)/ float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W, t=t: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads

if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    import os, sys
    sys.path.append(os.pardir)
    from dataset.mnist import load_mnist
    from two_layer_net import TwoLayerNet

    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    # mnistでロードしたデータをnumpyからcupyに変換
    x_train_cupy = cp.asarray(x_train)
    t_train_cupy = cp.asarray(t_train)

    train_loss_list = []
    # hyper parameter
    iters_num = 10
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

    network = TwoLayerNetGPU(784, 50, 10)

    iters_num = 1
    print("Start")
    start_time = time.time()

    start_time = time.time()
    for i in range(iters_num):
        print(f"trial: {i}")
        # ミニバッチの取得
        batch_mask = cp.random.choice(train_size, batch_size)

        x_batch = x_train_cupy[batch_mask]
        t_batch = t_train_cupy[batch_mask]

        # calculate gradient
        grad = network.numerical_gradient(x_batch, t_batch)

        # update parameter
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]

        # record
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
    end_time = time.time()
    print(f"time: {end_time-start_time}")   # 49.85s
    print(train_loss_list)
    # x = cp.arange(0, iters_num, 1)
    # plt.plot(train_loss_list)
    # plt.show()