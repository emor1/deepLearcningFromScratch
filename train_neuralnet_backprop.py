"""_summary_
誤差逆伝播法を使ったニューラルネットワークの学習
"""

import numpy as np
from dataset.mnist import load_mnist
from two_layer_net_backp import TwoLayerNet

from PIL import Image

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_eppoch = max(train_size/batch_size, 1)

print("Start")


for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.gradient(x_batch, t_batch)

    # 更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # print(network.accuracy(x_train, t_train))
    if i % iter_per_eppoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)


img_num = 12

print(t_test[img_num])
pil_img = Image.fromarray(np.uint(x_test[img_num].reshape(28,28)*255))
pil_img.show()

prediction = np.argmax(network.predict([x_test[img_num]]))
print("predict1: ", prediction)


img_num = 345

pil_img = Image.fromarray(np.uint(x_test[img_num].reshape(28,28)*255))
pil_img.show()

prediction = np.argmax(network.predict([x_test[img_num]]))
print("predict2: ", prediction)


print("Done!")
