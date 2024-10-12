from util import *
import json
from ch5_layer.layers import *
from collections import OrderedDict


class Convolution:
    def __init__(self, w, b, stride=1, pad=0):
        self.W = w
        self.b = b
        self.stride = stride
        self.pad = pad

        # 中間データ
        self.x = None
        self.col = None
        self.col_w = None

        # 重みバイアスの勾配
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2*self.pad - FH)/ self.stride)
        out_w = int(1 + (W + 2*self.pad - FW)/ self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_w = self.W.reshape(FN, -1).T # フィルターの展開
        out = np.dot(col, col_w) + self.b

        out = out.reshape(N, out_h, out_w, -1).transpose(0,3,1,2)
        self.x = x
        self.col = col
        self.col_w = col_w
        return out

    # 参考：https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/common/layers.py
    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_w.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)


class Pooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx


class SimpleConvolutionNet:
    """
    Parameters
    ----------
    input_dim : 入力データの（チャンネル、高さ、幅）の次元
    conv__param : 畳み込みそうのハイパーパラメータ
        - filter_num : フィルターの数
        - filter_size : フィルターのサイズ
        - stride :
        - pad: 
    hidden_size : 隠れ層の全結合のニューロンの数
    output_size : 出力そうのニューロンの数
    weight_init_std : 初期化の重みの標準偏差
    -------
    """
    def __init__(self, input_dim=(1,28.28), conv__param={'filter_num':30, 'filter_size': 5, 'stride': 1, 'pad':0}, hidden_size=100, output_size=10, weight_init_std=0.01):
        filter_num = conv__param['filter_num']
        filter_size = conv__param['filter_size']
        filter_pad = conv__param['pad']
        filter_stride = conv__param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))


        # 重みパラメータの初期化
        self.params = {}
        self.params['W1'] = weight_init_std* np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)

        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)

        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], conv__param['stride'], conv__param['pad'])
        self.layers['Relu1'] = Relu()

        self.layers['Pool'] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        self.last_layer = SoftMaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self,x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def gradient(self,x, t):
        """誤差逆伝播

        Parameters
        ----------
        x : 入力データ
        y : 教師データ
        ----------
        """
        # forward
        self.loss(x,t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db

        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db

        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db

        return grads

    # パラメータの保存
    def save_params(self, json_file):
        network_dict = {}
        for key in ('W1', 'b1', 'W2', 'b2','W3','b3'):
            param = self.params[key]
            network_dict[key] = param.tolist()

        with open(json_file, 'w') as f:
            json.dump(network_dict, f, indent=2)

        print('model saved')
