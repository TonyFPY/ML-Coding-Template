

import numpy as np

def conv1d(x, kernel, stride=1, padding=0):
    L = x.shape[0]
    K = kernel.shape[0]

    x = np.pad(x, (padding, padding))
    LO = (L + 2 * padding - K) // stride + 1

    out = np.zeros(LO)

    for i in range(LO):
        start = i * stride
        out[i] = np.sum(x[start:start + K] * kernel)

    return out


def conv2d(x, kernel, stride=1, padding=0):
    H, W = x.shape
    KH, KW = kernel.shape

    x = np.pad(x, ((padding, padding), (padding, padding)))

    HO = (H + 2 * padding - KH) // stride + 1
    WO = (W + 2 * padding - KW) // stride + 1

    out = np.zeros((HO, WO))

    for i in range(HO):
        for j in range(WO):
            row = i * stride
            col = j * stride

            window = x[row:row + KH, col:col + KW]
            out[i, j] = np.sum(window * kernel)

    return out


# Conv1D
x1 = np.array([1., 2., 3., 4., 5.])
k1 = np.array([1., 1., 1.])

print(conv1d(x1, k1, stride=2, padding=1))
# [3. 9. 9.]


# Conv2D
x2 = np.array([
    [1., 2., 3.],
    [4., 5., 6.],
    [7., 8., 9.]
])
k2 = np.ones((2, 2))

print(conv2d(x2, k2, stride=2, padding=1))
# [[ 1.  5.]
#  [11. 28.]]


