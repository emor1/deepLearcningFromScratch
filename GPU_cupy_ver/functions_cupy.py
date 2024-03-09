import cupy as cp


#---------------活性化関数---------------

# ステップ関数の実装
def step_function(x):
    y = x >0
    return y.astype(cp.int)

# シグモイド関数
def sigmoid(x):
    return 1/(1+cp.exp(-x))

# ReLU関数
def relu(x):
    return cp.maximum(0, x)

#---------------出力層---------------

#  恒等関数
def identity_function(x):
    return x

# ソフトマックス関数
def softmax(a):
    c = cp.max(a)
    exp_a = cp.exp(a-c)
    sum_exp_a = cp.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
