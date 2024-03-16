import numpy as np

# ReLU
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x<=0)
        # maskはboolのnumpy配列で、順伝播の入力のxの要素で0以下をtrue, 0より大きい要素をfalseとして保持
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1/(1+np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout*(1.0-self.out) * self.out
        return dx

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self,x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx

class SoftMaxWithLoss:
    def __init__(self):
        self.loss = None # 損失
        self.y = None # 出力
        self.t = None # 教師データ

    def forward(self, x, t):
        self.t = t
        self.y = self.softmax(x)
        self.loss = self.cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout =1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t)/batch_size
        return dx

    def cross_entropy_error(self, y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, t.size)

        batch_size = y.shape[0]
        return -np.sum(t*np.log(y+ 1e-7))/batch_size

    # ソフトマックス関数
    def softmax(self, a):
        c = np.max(a)
        exp_a = np.exp(a-c)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y


if __name__ == "__main__":
    x = np.array([[1.0, -0.5],[-2.0, 3.0]])
    print(x)
    mask = (x<=0)
    print(mask)

