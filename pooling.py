'''Implement 2D max pooling using NumPy and nested loops.'''


import numpy as np

def pool2d(x, kernel_size=2, stride=2, type="max"):
    # x: (H, W)
    H, W = x.shape
    K = kernel_size

    HO = (H - K) // stride + 1
    WO = (W - K) // stride + 1

    out = np.zeros((HO, WO))

    for i in range(HO):
        for j in range(WO):
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


x = np.array([
    [1., 2., 3., 4.],
    [5., 6., 7., 8.],
    [9., 10., 11., 12.],
    [13., 14., 15., 16.]
])

print(pool2d(x, kernel_size=2, stride=2, type="max"))
# [[ 6.  8.]
#  [14. 16.]]

print(pool2d(x, kernel_size=2, stride=2, type="avg"))
# [[ 3.5  5.5]
#  [11.5 13.5]]

x = np.array([
    [1., 2., 3.],
    [4., 5., 6.],
    [7., 8., 9.]
])

print(pool2d(x, kernel_size=2, stride=1, type="max"))
# [[5. 6.]
#  [8. 9.]]

'''
Part C — Interview follow-ups (5 minutes)
Be prepared to explain:
1. Max pooling vs. average pooling: How do their outputs differ, and what information does each preserve?
2. Overlapping windows: What happens when stride < kernel_size?
3. Backpropagation: How does the gradient pass through max pooling? What happens when multiple elements share the maximum?
4. Complexity: Why is the naïve implementation \(O(H_{\mathrm{out}}W_{\mathrm{out}}K^2)\)?
5. Multi-channel input: How would you extend pooling to (C, H, W) without mixing channels?
'''