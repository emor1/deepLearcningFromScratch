import numpy as np

"""_summary_
    参考：https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/common/util.py

"""

def sooth_curve(x):
    # 損失関数のグラフをなめらかにする
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    input_data : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
    filter_h : フィルターの高さ
    filter_w : フィルターの幅
    stride : ストライド
    pad : パディング

    Returns
    -------
    col : 2次元配列
    """

    N, C, H, W = input_data.shape
    out_h = (H+2*pad - filter_h)//stride + 1
    out_w = (W+2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad,pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:,:,y,x,:,:,] = img[:,:,y:y_max, x:x_max:stride]

    col = col.transpose(0,4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col