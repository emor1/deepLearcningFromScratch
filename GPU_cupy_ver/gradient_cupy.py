# ch4_loss.pyをcupyに書き換え

import cupy as cp

# バッチ対応版交差エントロピー誤差
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -cp.sum(t*cp.log(y+ 1e-7))/batch_size

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = cp.zeros_like(x)

    for idx in cp.ndindex(x.shape):
        # idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)

        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 値を元に戻す

    # grad = cp.gradient(f=f,)
    return grad