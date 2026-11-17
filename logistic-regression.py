'''
Part A — Implement two functions
Interview prompt:

You are building a binary classifier. Implement the following two functions using NumPy only.

1. stable_sigmoid(z): Accepts a NumPy array of arbitrary shape and returns elementwise sigmoid probabilities. Your implementation must avoid numerical overflow for very large positive or negative inputs.

2. binary_cross_entropy_with_logits(z, y): Accepts a one-dimensional NumPy array of logits and a matching array of binary targets (0 or 1). Return the mean binary cross-entropy loss as a scalar. Your implementation must remain numerically stable for extreme logits.

Constraints:
- Use NumPy only; no PyTorch, SciPy, or sklearn.
- Do not clip probabilities as a substitute for a stable logits-based loss.
- Use vectorized operations rather than Python loops.
- You may assume targets have valid values and matching shapes.
'''

import numpy as np

def stable_sigmoid(z):
    # TODO: Implement a numerically stable sigmoid
    
    z = np.asarray(z, dtype=float)
    ans = np.empty_like(z)

    pos = z >= 0
    ans[pos] = 1 / (1 + np.exp(-z[pos]))

    neg = z < 0
    ans[neg] = np.exp(z[neg]) / (1 + np.exp(z[neg]))

    return ans

def binary_cross_entropy_with_logits(z, y):
    # TODO: Return the mean BCE loss
    '''
    In PyTorch, it is torch.nn.functional.binary_cross_entropy_with_logits
    p = stable_sigmoid(z)
    l = - (y * np.log(p) + (1 - y) * np.log(1 - p))

    we can transform the standard binary cross-entropy loss into a mathematically equivalent expression
    l = np.log(1 + np.exp(z)) - y * z
    Here np.exp(z) can overflow when z is very large and postive
    we have the following equation to resitrict 0 < np.exp(-np.abs(z)) < 1
    '''
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)

    loss = np.maximum(z, 0) - z * y + np.log(1 + np.exp(-np.abs(z))) # np.maximum(z, 0) — Elementwise comparison

    return np.mean(loss)
    

z = np.array([0.0, 2.0, -2.0])
print(stable_sigmoid(z))

z = np.array([1000.0, -1000.0])
y = np.array([1.0, 0.0])
print(stable_sigmoid(z))
print(binary_cross_entropy_with_logits(z, y))

z = np.zeros(4)
y = np.array([1., 0., 1., 0.])
print(binary_cross_entropy_with_logits(z, y))

'''
Part B — Logistic Regression Gradients and SGD
Suppose binary classification logits are computed using

\\[ z=Xw+b \\]

where \\(X\in\mathbb{R}^{B\times D}\\), \\(w\in\mathbb{R}^{D}\\), and \\(b\\) is a scalar.

The interviewer will ask you to:

1. Derive the analytical gradients \\(\nabla_w L\\) and \\(\partial L/\partial b\\).
2. Implement one SGD update using those gradients.
3. Explain why computing BCE directly from logits is more stable than applying sigmoid followed by logarithms.
'''


def logistic_regression_step(X, y, w, b, lr):
    # X: (B, D)
    # y: (B,)
    # w: (D,)
    # b: scalar

    B = X.shape[0]

    # Forward pass
    z = X @ w + b
    p = stable_sigmoid(z)

    # BCE
    loss = binary_cross_entropy_with_logits(z, y)

    # Backward loss
    dz = (p - y) / B        # (B,)
    dw = X.T @ dz          # (D,)
    db = np.sum(dz)        # scalar

    # SGD update
    w = w - lr * dw
    b = b - lr * db

    return w, b, loss


X = np.array([
    [1., 2.],
    [2., 1.],
    [-1., -2.],
    [-2., -1.]
])

y = np.array([1., 1., 0., 0.])

w = np.zeros(2)
b = 0.0
lr = 0.1

print("Loss before:", binary_cross_entropy_with_logits(X @ w + b, y))

w, b, loss = logistic_regression_step(X, y, w, b, lr)

print("Updated weights:", w)
print("Updated bias:", b)
print("Loss after:", binary_cross_entropy_with_logits(X @ w + b, y))