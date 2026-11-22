# Image Interpolation with NumPy

Interpolation estimates an image value between pixels. This guide follows `interpolation.py`: bilinear interpolation blends four nearby pixels, bilinear resize maps output pixels back to input coordinates, and nearest-neighbor interpolation selects one pixel.

## Notation

| Symbol | Meaning |
| :--- | :--- |
| $X$ | Input image with shape $(H,W)$ |
| $(r,c)$ | Fractional input coordinate: row first, then column |
| $(r_0,c_0)$ | Top-left integer pixel coordinate |
| $(r_1,c_1)$ | Bottom-right coordinate of the surrounding pixel cell |
| $\alpha$ | Fraction of the distance from row $r_0$ toward $r_1$ |
| $\beta$ | Fraction of the distance from column $c_0$ toward $c_1$ |

## Bilinear Interpolation

The query point lies between four neighboring pixels. First blend horizontally along the top and bottom edges, then blend those two results vertically.

```text
                    column
                 c0       c1
              ┌─────────────┐
row r0        │ A ── β ── B │
              │     Q       │   α is the fraction down from r0 to r1
row r1        │ C ── β ── D │
              └─────────────┘
```

Here $Q=(r_0+\alpha,c_0+\beta)$, with $0\le\alpha,\beta\le1$. The pixel weights are:

| Pixel | Weight |
| :--- | :--- |
| $A=X_{r_0,c_0}$ | $(1-\alpha)(1-\beta)$ |
| $B=X_{r_0,c_1}$ | $(1-\alpha)\beta$ |
| $C=X_{r_1,c_0}$ | $\alpha(1-\beta)$ |
| $D=X_{r_1,c_1}$ | $\alpha\beta$ |

Equivalently, interpolate across each row, then between the rows:

$$
I_{top}=(1-\beta)A+\beta B,
\qquad
I_{bottom}=(1-\beta)C+\beta D,
$$

$$
I(r,c)=(1-\alpha)I_{top}+\alpha I_{bottom}.
$$

The weights sum to one, so the result is a weighted average of the four pixel values.

```python
import numpy as np


def bilinear_interpolate(x, row, col):
    H, W = x.shape

    # Keep coordinates on the image; outside points use the nearest edge.
    row = np.clip(row, 0, H - 1)
    col = np.clip(col, 0, W - 1)

    r0 = int(np.floor(row))
    c0 = int(np.floor(col))
    r1 = min(r0 + 1, H - 1)
    c1 = min(c0 + 1, W - 1)

    alpha = row - r0
    beta = col - c0

    top = (1 - beta) * x[r0, c0] + beta * x[r0, c1]
    bottom = (1 - beta) * x[r1, c0] + beta * x[r1, c1]
    return (1 - alpha) * top + alpha * bottom
```

### Worked example

For

```text
X = [[10, 20],
     [30, 40]]
```

at `(row, col) = (0.25, 0.75)`, the point is one-quarter of the way down and three-quarters across. The top blend is $17.5$, the bottom blend is $37.5$, and the final vertical blend is $22.5$:

```python
x = np.array([[10., 20.], [30., 40.]])
print(bilinear_interpolate(x, 0.25, 0.75))
# 22.5
```

At `(0.5, 0.5)`, all four weights are $1/4$, so the center value is `25.0`. At an exact integer coordinate, the result is that pixel's value.

## Bilinear Image Resize

For each output pixel, `bilinear_resize` maps its coordinate back to the input image. This source aligns the first and last output pixels with the first and last input pixels:

$$
r=i\frac{H-1}{new\_H-1},
\qquad
c=j\frac{W-1}{new\_W-1}.
$$

When an output dimension is one, the source uses input coordinate zero for that axis.

```python
def bilinear_resize(x, new_H, new_W):
    H, W = x.shape
    output = np.zeros((new_H, new_W))

    for i in range(new_H):
        for j in range(new_W):
            row = i * (H - 1) / (new_H - 1) if new_H > 1 else 0
            col = j * (W - 1) / (new_W - 1) if new_W > 1 else 0
            output[i, j] = bilinear_interpolate(x, row, col)

    return output
```

Resizing the 2×2 example to 3×3 maps rows and columns to `[0, 0.5, 1]`:

```text
[[10.0, 15.0, 20.0],
 [20.0, 25.0, 30.0],
 [30.0, 35.0, 40.0]]
```

The loops visit every output pixel. Each pixel takes constant work, so resize costs $O(new\_H\,new\_W)$.

## Nearest-Neighbor Interpolation

Nearest-neighbor interpolation rounds the fractional coordinate to one pixel, then clips it to the image boundary. The source rounds half values upward using `floor(value + 0.5)`:

```python
def nearest_interpolate(x, row, col):
    H, W = x.shape

    row = int(np.floor(row + 0.5))
    col = int(np.floor(col + 0.5))

    row = np.clip(row, 0, H - 1)
    col = np.clip(col, 0, W - 1)
    return x[row, col]
```

For the 2×2 example, `(0.25, 0.75)` selects pixel `(0, 1)` with value `20`; bilinear interpolation instead blends the four neighbors and returns `22.5`. Nearest-neighbor preserves original pixel values, while bilinear produces smoother transitions.

## Boundary and Numerical Notes

The bilinear function clips coordinates before selecting neighbors. For example, `(-0.5, 1.5)` in a 2×2 image becomes `(0, 1)`, returning `20.0`; it does not extrapolate beyond the image. At the last row or column, the two neighbor indices can be equal, which still returns the edge pixel correctly.

For ordinary image values, no special numerical stabilization is needed. The functions assume a nonempty 2D image and positive output dimensions for resizing.
