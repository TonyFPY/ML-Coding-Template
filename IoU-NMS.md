# IoU and Non-Maximum Suppression

Object detectors can return several boxes around the same object. Intersection over Union (IoU) measures how much two boxes overlap; greedy Non-Maximum Suppression (NMS) uses that score to keep strong detections and remove redundant ones.

## Notation

Boxes use continuous corner coordinates `(x1, y1, x2, y2)`: top-left followed by bottom-right.

| Symbol | Meaning |
| :--- | :--- |
| $A,B$ | Two boxes |
| $|A|,|B|$ | Box areas |
| $|A\cap B|$ | Intersection area |
| $|A\cup B|$ | Union area |
| $s_i$ | Confidence score for box $i$ |
| $\tau$ | IoU suppression threshold |

## 1. Intersection over Union

The intersection rectangle is bounded by the larger left/top coordinates and the smaller right/bottom coordinates:

$$
x_L=\max(x_{1,A},x_{1,B}),\quad y_T=\max(y_{1,A},y_{1,B}),
\qquad
x_R=\min(x_{2,A},x_{2,B}),\quad y_B=\min(y_{2,A},y_{2,B}).
$$

If the boxes do not overlap on an axis, that intersection dimension is zero:

$$
w_I=\max(0,x_R-x_L),\quad h_I=\max(0,y_B-y_T),\quad |A\cap B|=w_Ih_I.
$$

Then subtract the intersection once when computing the union:

$$
\operatorname{IoU}(A,B)=\frac{|A\cap B|}{|A|+|B|-|A\cap B|}.
$$

### Picture: partial overlap

For $A=[0,0,2,2]$ and $B=[1,1,3,3]$, both boxes have area $4$. Their shared width and height are each $1$:

```text
x coordinates:
A  [0----------------2]
B           [1----------------3]
             [1---2]  shared width = 1

y coordinates:
A  [0----------------2]
B           [1----------------3]
             [1---2]  shared height = 1
```

So the intersection is $1$, the union is $4+4-1=7$, and IoU is $1/7\approx0.1429$. Subtracting the intersection matters because it was counted in both box areas.

```python
def iou(box1, box2):
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])

    intersection_width = max(0, ix2 - ix1)
    intersection_height = max(0, iy2 - iy1)
    intersection = intersection_width * intersection_height

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0
    return intersection / union
```

This uses continuous box coordinates, so there is no `+1` in width or height. Boxes are assumed to have ordered corners. A zero-area union returns `0.0`, matching the source's convention for degenerate boxes.

### Vectorized IoU

For one box and an array of $N$ boxes, each coordinate operation broadcasts across all rows. The resulting intersection, union, and IoU arrays have shape `(N,)`.

```python
import numpy as np


def iou_vectorized(box, boxes):
    # box:   (4,)
    # boxes: (N, 4)
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0, ix2 - ix1)
    inter_h = np.maximum(0, iy2 - iy1)
    inter_area = inter_w * inter_h

    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter_area

    return np.where(union > 0, inter_area / union, 0.0)
```

For `box = [0, 0, 2, 2]` and boxes `[[0,0,2,2], [1,1,3,3], [3,3,5,5], [0,0,1,1]]`, the result is `[1, 1/7, 0, 1/4]`.

## 2. Greedy Non-Maximum Suppression

NMS sorts boxes by descending confidence. It repeatedly keeps the highest-scoring remaining box and removes later boxes whose IoU with it is greater than the threshold.

```text
Sorted candidates:   box 0 (.90)   box 1 (.80)   box 2 (.70)
                         │
Keep box 0 ──────────────┼── compare with boxes 1 and 2
                         ├── box 1: IoU > τ  → suppress
                         └── box 2: IoU ≤ τ  → keep for next round

Returned original indices: [0, 2]
```

The comparison is strict: a box is suppressed only when `IoU > threshold`; equality is retained.

```python
def nms(boxes, scores, threshold=0.5):
    order = np.argsort(-scores)  # descending score order
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        remaining = order[1:]
        ious = iou_vectorized(boxes[i], boxes[remaining])
        order = remaining[ious <= threshold]

    return keep
```

`keep` contains indices into the original `boxes` and is ordered by the greedy selection. For the source's first NMS example, boxes 0 and 1 overlap by about `0.681`, which is above `0.5`; box 0 is kept, box 1 is suppressed, and disjoint box 2 is kept. The result is `[0, 2]`.

### Class-aware NMS

Class-agnostic NMS can suppress boxes for different objects if they overlap. The source handles this by running NMS separately for each label, mapping local indices back to the original arrays, then sorting retained indices by score:

```python
def class_aware_nms(boxes, scores, labels, threshold=0.5):
    keep = []

    for cls in np.unique(labels):
        indices = np.where(labels == cls)[0]
        selected = nms(boxes[indices], scores[indices], threshold)
        keep.extend(indices[selected].tolist())

    return sorted(keep, key=lambda i: scores[i], reverse=True)
```

## Numerical Edge Case

The scalar `iou` checks for a nonpositive union before dividing. In the vectorized function, `np.where` evaluates `inter_area / union` before selecting the zero-union result, which can emit a divide-by-zero warning for degenerate boxes. A guarded division avoids evaluating those entries:

```python
ious = np.zeros_like(union, dtype=float)
np.divide(inter_area, union, out=ious, where=union > 0)
return ious
```

This keeps the source's convention of returning zero when the union is nonpositive.

## Interview Explanations

- **Why sort by score?** Greedy NMS gives priority to the most confident detection, then compares lower-scoring candidates against boxes already kept.
- **Why vectorize IoU?** One call compares a selected box with every remaining box using NumPy arrays of shape `(N,)`, avoiding a Python loop over candidates for the overlap calculation.
- **What is the worst-case cost?** Sorting costs $O(N\log N)$. In the worst case, each selected box is compared with many remaining boxes, giving $O(N^2)$ IoU work.
- **What does the threshold control?** A lower threshold removes more overlapping boxes; a higher threshold allows more overlap. The source keeps boxes with IoU equal to the threshold.
