# 2D Pooling with NumPy

Pooling summarizes each spatial window of a 2D input. The source implements square max pooling and average pooling with no padding.

## Notation

| Symbol | Shape | Meaning |
| :--- | :--- | :--- |
| $X$ | $(H,W)$ | Input matrix |
| $K$ | scalar | Square window size; window shape is $(K,K)$ |
| $s$ | scalar | Stride between window starts |
| $Y$ | $(H_{out},W_{out})$ | Pooled output |

Because the implementation uses valid windows without padding:

$$
H_{out}=\left\lfloor\frac{H-K}{s}\right\rfloor+1,
\qquad
W_{out}=\left\lfloor\frac{W-K}{s}\right\rfloor+1.
$$

## Version 1 — Max and Average Pooling

For output position $(i,j)$, the window starts at row $is$ and column $js$. Max pooling keeps the largest value:

$$
Y_{ij}^{\max}=\max_{0\le u,v<K}X_{is+u,js+v}.
$$

Average pooling returns the mean of the $K^2$ values:

$$
Y_{ij}^{\mathrm{avg}}=\frac{1}{K^2}\sum_{u=0}^{K-1}\sum_{v=0}^{K-1}X_{is+u,js+v}.
$$

```python
import numpy as np


def pool2d(x, kernel_size=2, stride=2, type="max"):
    H, W = x.shape
    K = kernel_size
    out_height = (H - K) // stride + 1
    out_width = (W - K) // stride + 1

    out = np.zeros((out_height, out_width))
    for i in range(out_height):
        for j in range(out_width):
            row = i * stride
            col = j * stride
            window = x[row:row + K, col:col + K]

            if type == "max":
                out[i, j] = np.max(window)
            elif type == "avg":
                out[i, j] = np.mean(window)
            else:
                raise ValueError("type must be 'max' or 'avg'")

    return out
```

The output indices `i` and `j` count windows. Multiplying them by `stride` converts them to input coordinates. If `stride < kernel_size`, adjacent windows overlap; if equal, the windows do not overlap.

Example from the source:

```python
x = np.arange(1., 17.).reshape(4, 4)
print(pool2d(x, kernel_size=2, stride=2, type="max"))
# [[ 6.  8.]
#  [14. 16.]]

print(pool2d(x, kernel_size=2, stride=2, type="avg"))
# [[ 3.5  5.5]
#  [11.5 13.5]]
```

With stride 1, the windows overlap:

```python
x = np.arange(1., 10.).reshape(3, 3)
print(pool2d(x, kernel_size=2, stride=1, type="max"))
# [[5. 6.]
#  [8. 9.]]
```

## Version 2 — Numerical Notes

Pooling does not use exponentials or a normalization denominator, so it has no special stability transform here. Average pooling can still accumulate floating-point rounding or overflow for extreme values; the examples use ordinary floating-point inputs.

## Interview Follow-ups

1. **Max vs. average:** Max pooling preserves whether a strong value appeared in a window and discards the other values. Average pooling preserves the window's overall level but smooths sharp peaks.

2. **Overlapping windows:** When `stride < kernel_size`, windows share input values. An input can therefore affect several output positions. The output size still follows the floor formulas above.

3. **Backpropagation through max pooling:** The gradient for an output is sent to the input value that won the maximum; gradients from overlapping windows add together. This function is forward-only and does not save the winning indices. If a window has tied maxima, a backward implementation must choose a tie rule, such as routing to one maximum or splitting the gradient among them.

4. **Complexity:** There are $H_{out}W_{out}$ windows, each containing $K^2$ values. Scanning each window costs $O(H_{out}W_{out}K^2)$.

5. **Multiple channels:** For input shape $(C,H,W)$, apply the same pooling operation independently to each $X_c$ plane. Return shape $(C,H_{out},W_{out})$; do not pool across the channel axis.
