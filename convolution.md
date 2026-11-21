# Conv1D and Conv2D in NumPy

These interview implementations compute deep-learning convolution with NumPy loops. More precisely, they compute cross-correlation: each input window is multiplied by the kernel without flipping it.

## Notation

| Input | Kernel | Output |
| :--- | :--- | :--- |
| 1D: $x\in\mathbb{R}^{L}$ | $k\in\mathbb{R}^{K}$ | $y\in\mathbb{R}^{L_{out}}$ |
| 2D: $X\in\mathbb{R}^{H\times W}$ | $K\in\mathbb{R}^{K_H\times K_W}$ | $Y\in\mathbb{R}^{H_{out}\times W_{out}}$ |

For stride $s$ and zero padding $p$ on each side:

$$
L_{out}=\left\lfloor\frac{L+2p-K}{s}\right\rfloor+1,
\quad
H_{out}=\left\lfloor\frac{H+2p-K_H}{s}\right\rfloor+1,
\quad
W_{out}=\left\lfloor\frac{W+2p-K_W}{s}\right\rfloor+1.
$$

Assume positive integer stride, nonnegative integer padding, and kernels that fit inside the padded input.

## Version 1 — Basic Implementation

Let $\tilde{x}$ and $\tilde{X}$ be the inputs after zero padding. The 1D output is:

$$
y_i=\sum_{r=0}^{K-1}\tilde{x}_{is+r}k_r,
\qquad 0\le i<L_{out}.
$$

The 2D output is:

$$
Y_{i,j}=\sum_{u=0}^{K_H-1}\sum_{v=0}^{K_W-1}
\tilde{X}_{is+u,js+v}K_{u,v}.
$$

```python
import numpy as np


def conv1d(x, kernel, stride=1, padding=0):
    L, K = x.shape[0], kernel.shape[0]
    x = np.pad(x, (padding, padding))
    output_length = (L + 2 * padding - K) // stride + 1

    out = np.zeros(output_length)
    for i in range(output_length):
        start = i * stride
        out[i] = np.sum(x[start:start + K] * kernel)

    return out


def conv2d(x, kernel, stride=1, padding=0):
    H, W = x.shape
    KH, KW = kernel.shape
    x = np.pad(x, ((padding, padding), (padding, padding)))

    out_height = (H + 2 * padding - KH) // stride + 1
    out_width = (W + 2 * padding - KW) // stride + 1
    out = np.zeros((out_height, out_width))

    for i in range(out_height):
        for j in range(out_width):
            row, col = i * stride, j * stride
            window = x[row:row + KH, col:col + KW]
            out[i, j] = np.sum(window * kernel)

    return out
```

The output index `i` is not the input start: the input window begins at `i * stride`. In 2D, use `i * stride` for rows and `j * stride` for columns. The output width formula uses input width $W$.

Example with stride and padding:

```python
x = np.array([1., 2., 3., 4., 5.])
print(conv1d(x, np.ones(3), stride=2, padding=1))
# [3. 9. 9.]

X = np.arange(1., 10.).reshape(3, 3)
print(conv2d(X, np.ones((2, 2)), stride=2, padding=1))
# [[ 1.  5.]
#  [11. 28.]]
```

Strict mathematical convolution flips the kernel first. This code follows the cross-correlation convention used by common deep-learning layers.

## Version 2 — Numerical Stability

There is no softmax-style shift that stabilizes arbitrary convolution inputs. These functions multiply and sum directly; use floating-point inputs because integer multiplication can overflow before the result is stored. Very large floating-point products or sums can still overflow.

The direct implementation costs $O(L_{out}K)$ for 1D and $O(H_{out}W_{out}K_HK_W)$ for 2D.
