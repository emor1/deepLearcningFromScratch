import numpy as np


#---------------活性化関数---------------

# ステップ関数の実装
def step_function(x):
    y = x >0
    return y.astype(np.int)

# シグモイド関数
def sigmoid(x):
    return 1/(1+np.exp(-x))

# ReLU関数
def relu(x):
    return np.maximum(0, x)

def init_network():
    # 主にバイアスと重みの初期化を行う
    network = {} # dictionary
    network['W1'] = np.array([[0.1, 0.3, 0.5],[0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])

    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])

    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    return y


#---------------出力層---------------

#  恒等関数
def identity_function(x):
    return x

# ソフトマックス関数
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

