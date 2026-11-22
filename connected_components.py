'''
Connected-component labeling is a common image-processing problem that combines grid traversal, BFS/DFS, and neighborhood definitions. It is closely related to LeetCode 200 (Number of Islands).
'''




import numpy as np
from collections import deque

def connected_components(x, connectivity=4):
    H, W = x.shape
    labels = np.zeros((H, W), dtype=int)
    count = 0

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    if connectivity == 8:
        directions += [(-1, -1), (-1, 1),
                       (1, -1), (1, 1)]

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

