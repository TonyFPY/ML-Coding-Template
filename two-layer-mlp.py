'''
Implement a two-layer MLP with forward and backward passes. 
Use only NumPy

x -> linear layer -> Relu -> linear layer -< softmax -> cross-entropy loss
'''


import numpy as np

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def ce_loss(y, p):
    B = y.shape[0]
    loss = 0.0

    for i in range(B):
        loss -= np.log(p[i, y[i]])

    return loss / B

def mlp_forward(x, w1, b1, w2, b2, y):
    z1 = x @ w1 + b1          # (B, H)
    a1 = relu(z1)             # (B, H)
    z2 = a1 @ w2 + b2         # (B, C)

    p = softmax(z2)
    loss = ce_loss(y, p)

    cache = (x, y, w2, z1, a1, p)

    return loss, cache

def mlp_backward(cache):
    x, y, w2, z1, a1, p = cache
    B = x.shape[0]

    # Softmax + cross-entropy
    one_hot = np.zeros_like(p)

    for i in range(B):
        one_hot[i, y[i]] = 1

    dl_dz2 = (p - one_hot) / B

    # Linear layer 2
    dl_dw2 = a1.T @ dl_dz2
    dl_db2 = np.sum(dl_dz2, axis=0)

    # ReLU backward
    dl_da1 = dl_dz2 @ w2.T
    dl_dz1 = dl_da1 * (z1 > 0)

    # Linear layer 1
    dl_dw1 = x.T @ dl_dz1
    dl_db1 = np.sum(dl_dz1, axis=0)

    return dl_dw1, dl_db1, dl_dw2, dl_db2


def sgd(w1, b1, w2, b2, grads, lr):
    dw1, db1, dw2, db2 = grads

    w1 -= lr * dw1
    b1 -= lr * db1
    w2 -= lr * dw2
    b2 -= lr * db2

    return w1, b1, w2, b2


# Demo dimensions: batch B=2, input D=2, hidden H=3, classes C=2.
X = np.array([[1., 2.],
              [3., 4.]]) # 2 * 2
W1 = np.array([[0.1, 0.2, 0.3],
               [0.4, 0.5, 0.6]]) # 2 * 3
b1 = np.zeros(3)
W2 = np.array([[0.1, -0.1],
               [0.2, -0.2],
               [0.3, -0.3]])
b2 = np.zeros(2)
y = np.array([0, 1])

lr = 0.1

# Forward
loss, cache = mlp_forward(X, W1, b1, W2, b2, y)
print("Before:", loss)

# Backward
grads = mlp_backward(cache)

# Update
W1, b1, W2, b2 = sgd(
    W1, b1, W2, b2, grads, lr
)

# Forward again
new_loss, _ = mlp_forward(X, W1, b1, W2, b2, y)
print("After:", new_loss)