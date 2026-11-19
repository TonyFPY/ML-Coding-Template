
'''
X -> Linear -> ReLU -> Linear -> softmax -> cross-entropy loss
'''

import numpy as np
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# x (B, D)
# w1 (D, H) b1(H,)
# w2 (H, C) b2(C, )
# y (B, )
def mlp_forward(x, w1, b1, w2, b2, y):
    z1 = x @ w1 + b1 
    a1 = relu(z1)
    z2 = a1 @ w2 + b2

    cache = (x, w1, b1, w2, b2, y, z1, a1, z2)
    return cache

def ce_loss(y, prob):
    rows = np.arange(y.shape[0])
    return -np.mean(np.log(prob[rows, y]))

def mlp_backward(cache):
    x, w1, b1, w2, b2, y, z1, a1, z2 = cache
    p = softmax(z2)
    loss = ce_loss(y, p)

    B = x.shape[0]
    one_hot = np.zeros_like(p)
    one_hot[np.arange(B), y] = 1
    dz2 = (p - one_hot) / B 

    dw2 = a1.T @ dz2 
    db2 = np.sum(dz2, axis=0)

    da1 = dz2 @ w2.T
    dz1 = da1 * (z1 > 0)

    dw1 = x.T @ dz1
    db1 = np.sum(dz1, axis=0)

    return dw2, db2, dw1, db1


