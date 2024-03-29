import numpy as np
import matplotlib.pyplot as plt

from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []

# hyper parameter
iters_num = 5
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=50, output_size= 10)

for i in range(iters_num):
    print(f"trial: {i}")
    # ミニバッチの取得
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # calculate gradient
    grad = network.numerical_gradient(x_batch, t_batch)

    # update parameter
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # record
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

print(train_loss_list)
x = np.arange(0, iters_num, 1)
plt.plot(train_loss_list)
plt.show()