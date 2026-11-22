# Connected Components in a Binary Grid

Connected-component labeling assigns one integer label to each group of connected foreground cells. This implementation treats cells equal to `1` as foreground, uses breadth-first search (BFS), and supports 4- or 8-neighbor connectivity.

## Notation

| Variable | Shape | Meaning |
| :--- | :--- | :--- |
| `x` | `(H, W)` | Input grid; `1` is foreground |
| `labels` | `(H, W)` | Component ID at each foreground cell; background stays `0` |
| `count` | scalar | Number of components found |
| `connectivity` | `4` or `8` | Which neighboring cells can connect |

Each cell is a graph node. Edges connect the neighbors shown below:

```text
4-connectivity                 8-connectivity

    ↑                               ↖ ↑ ↗
  ← · →                           ← · →
    ↓                               ↙ ↓ ↘
```

Diagonal cells form separate components with 4-connectivity, but one component with 8-connectivity:

```text
Input:             4-connected labels:    8-connected labels:
1 0                1 0                    1 0
0 1                0 2                    0 1
```

## BFS Solution

Scan the grid in row-major order. When an unlabeled foreground cell is found, give it a new component ID and explore every foreground cell reachable from it. Mark a cell when it enters the queue; this prevents it from being queued more than once.

```python
import numpy as np
from collections import deque


def connected_components(x, connectivity=4):
    H, W = x.shape
    labels = np.zeros((H, W), dtype=int)
    count = 0

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for i in range(H):
        for j in range(W):
            if x[i, j] == 1 and labels[i, j] == 0:
                count += 1
                queue = deque([(i, j)])
                labels[i, j] = count

                while queue:
                    r, c = queue.popleft()

                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if x[nr, nc] == 1 and labels[nr, nc] == 0:
                                labels[nr, nc] = count
                                queue.append((nr, nc))

    return labels, count
```

The neighbor offsets `(dr, dc)` say how to move from the current cell. The bounds check keeps those moves inside the grid. `labels == 0` means “not visited yet”; when a foreground cell is labeled, the algorithm will not visit it again.

## Worked Grid

```text
Input:                  4-connectivity labels:
1 1 0 0 1             1 1 0 0 2
0 1 0 0 0             0 1 0 0 0
0 0 1 1 0             0 0 3 3 0
1 0 0 1 0             4 0 0 3 0
```

The two groups that touch only diagonally remain separate, so the count is `4`. With 8-connectivity, the diagonal path joins the first and third groups:

```text
8-connectivity labels:
1 1 0 0 2
0 1 0 0 0
0 0 1 1 0
3 0 0 1 0
```

The count is then `3`. Label numbers depend on scan order; the important result is which cells share a label.

## Complexity and Input Behavior

The scan takes $O(HW)$ time. Each foreground cell is added to the queue at most once, and it checks at most eight neighbors. The label grid uses $O(HW)$ space; the queue can also grow to $O(HW)$ in the largest component.

The code uses exact equality with `1`; every other value is treated as background. It expects a 2D NumPy array and `connectivity` of `4` or `8`. No numerical-stability step is needed because the algorithm uses discrete labels rather than floating-point calculations.
