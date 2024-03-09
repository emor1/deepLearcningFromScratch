import numpy as np

# バッチ対応版交差エントロピー誤差
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t*np.log(y+ 1e-7))/batch_size

# def numerical_gradient(f, x):
#     h = 1e-4
#     grad = np.zeros_like(x) # xと同じ形状の配列を生成

#     for idx in range(x.size):
#         tmp_val = x[idx]
#         # f(x+h)
#         x[idx] = tmp_val + h
#         fxh1 = f(x)

#         # f(x-h)
#         x[idx] = tmp_val - h
#         fxh2 = f(x)

#         grad[idx] = (fxh1 - fxh2)/(2*h)
#         x[idx] = tmp_val

#     return grad


def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 値を元に戻す
        it.iternext()   
        
    return grad