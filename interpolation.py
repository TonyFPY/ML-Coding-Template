'''
Given a 2D image x of shape (H, W) and a fractional coordinate (row, col), estimate the pixel value using the four nearest surrounding pixels.

Implement the operation using NumPy without calling any built-in interpolation or image-resizing functions.
'''

import numpy as np

def bilinear_interpolate(x, row, col):
    """
    x: (H, W)
    row: fractional row coordinate
    col: fractional column coordinate

    Returns:
        interpolated scalar value
    """

    H, W = x.shape

    row = np.clip(row, 0, H - 1)
    col = np.clip(col, 0, W - 1)

    # get pixel coordinates of the four nearest surrounding pixels
    r0 = int(np.floor(row))
    c0 = int(np.floor(col))
    r1 = min(r0 + 1, H - 1)
    c1 = min(c0 + 1, W - 1)

    # calculate the fractional distances from the top-left pixel
    alpha = row - r0
    beta = col - c0

    # interpolate horizontally between the four surrounding pixels
    I_top = (1 - beta) * x[r0, c0] + beta * x[r0, c1]
    I_bottom = (1 - beta) * x[r1, c0] + beta * x[r1, c1]

    # interpolate vertically between the two horizontally interpolated values
    I = (1 - alpha) * I_top + alpha * I_bottom

    return I

x = np.array([
    [10., 20.],
    [30., 40.]
])

# Center
print(bilinear_interpolate(x, 0.5, 0.5))
# 25.0

# Fractional coordinate
print(bilinear_interpolate(x, 0.25, 0.75))
# 22.5

# Exact pixel
print(bilinear_interpolate(x, 1.0, 0.0))
# 30.0

# Outside boundary
print(bilinear_interpolate(x, -0.5, 1.5))
# 20.0


'''
Given an image of shape (H, W), resize it to (new_H, new_W) using your bilinear interpolation function.
'''

def bilinear_resize(x, new_H, new_W):
    H, W = x.shape

    output = np.zeros((new_H, new_W))

    for i in range(new_H):
        for j in range(new_W):

            # Map output coordinate to input coordinate
            row = i * (H - 1) / (new_H - 1) if new_H > 1 else 0
            col = j * (W - 1) / (new_W - 1) if new_W > 1 else 0

            # Interpolate
            output[i, j] = bilinear_interpolate(x, row, col)

    return output

x = np.array([
    [10., 20.],
    [30., 40.]
])

print(bilinear_resize(x, 3, 3))

'''
Nearest-neighbor interpolation
Instead of using four neighboring pixels, return the value of the single nearest pixel.
'''


def nearest_interpolate(x, row, col):
    H, W = x.shape

    row = int(np.floor(row + 0.5))
    col = int(np.floor(col + 0.5))

    row = np.clip(row, 0, H - 1)
    col = np.clip(col, 0, W - 1)

    return x[row, col]
