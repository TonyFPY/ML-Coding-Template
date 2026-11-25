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

## Follow-Up — Batched, Multi-Channel Convolution

The basic functions above take one single-channel input. These interview follow-ups extend them to a batch of inputs, multiple input channels, and multiple output filters. They still compute cross-correlation, with one shared stride and symmetric zero padding.

For Conv1D, `X` has shape $(B,C_{in},L)$ and `W` has shape $(C_{out},C_{in},K)$. The output shape is $(B,C_{out},L_{out})$, where

$$
L_{out}=\left\lfloor\frac{L+2p-K}{s}\right\rfloor+1.
$$

Each output sums over input channels and kernel positions:

$$
Y_{b,o,t}=\sum_{c=0}^{C_{in}-1}\sum_{r=0}^{K-1}
\widetilde{X}_{b,c,ts+r}W_{o,c,r}+b_o.
$$

For Conv2D, `X` has shape $(B,C_{in},H,W_{in})$ and `W` has shape $(C_{out},C_{in},K_h,K_w)$. The output shape is $(B,C_{out},H_{out},W_{out})$, with

$$
H_{out}=\left\lfloor\frac{H+2p-K_h}{s}\right\rfloor+1,
\qquad
W_{out}=\left\lfloor\frac{W_{in}+2p-K_w}{s}\right\rfloor+1.
$$

```python
import numpy as np


def conv1d(X, W, b=None, stride=1, padding=0):
    # X: (B, C_in, L); W: (C_out, C_in, K)
    B, C_in, L = X.shape
    C_out, _, K = W.shape

    X = np.pad(X, ((0, 0), (0, 0), (padding, padding)))
    L_out = (L + 2 * padding - K) // stride + 1
    out = np.zeros((B, C_out, L_out))

    for i in range(L_out):
        start = i * stride
        window = X[:, :, start:start + K]  # (B, C_in, K)
        out[:, :, i] = np.sum(
            window[:, None, :, :] * W[None, :, :, :],
            axis=(-2, -1),
        )

    if b is not None:
        out += b[None, :, None]

    return out


def conv2d(X, W, b=None, stride=1, padding=0):
    # X: (B, C_in, H, W_in); W: (C_out, C_in, K_h, K_w)
    B, C_in, H, W_in = X.shape
    C_out, _, K_h, K_w = W.shape

    X = np.pad(X, (
        (0, 0),
        (0, 0),
        (padding, padding),
        (padding, padding),
    ))

    H_out = (H + 2 * padding - K_h) // stride + 1
    W_out = (W_in + 2 * padding - K_w) // stride + 1
    out = np.zeros((B, C_out, H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride
            window = X[
                :, :, h_start:h_start + K_h, w_start:w_start + K_w
            ]  # (B, C_in, K_h, K_w)
            out[:, :, i, j] = np.sum(
                window[:, None, :, :, :] * W[None, :, :, :, :],
                axis=(-3, -2, -1),
            )

    if b is not None:
        out += b[None, :, None, None]

    return out
```

In each window, `window[:, None, ...]` inserts an output-filter axis. Broadcasting multiplies that window against every filter, and the sum reduces the input-channel and kernel axes. The optional bias broadcasts once per output channel across batch and spatial positions.
