# K-Nearest Neighbors with NumPy

K-Nearest Neighbors (KNN) predicts from nearby training examples. It has no separate fitting step: for each query, it measures distance to the training data, selects the nearest $K$ samples, then votes for classification or averages their targets for regression.

## Notation and Shapes

| Variable | Shape | Meaning |
| :--- | :--- | :--- |
| `X_train` | `(N, D)` | $N$ training examples with $D$ features |
| `y_train` | `(N,)` | Training class labels or target values |
| `X_test` | `(M, D)` | $M$ query examples |
| `distances` | `(M, N)` | Distance from each query to every training example |
| `indices` | `(M, K)` | Indices of the $K$ nearest training examples |

Euclidean distance between feature vectors $x$ and $y$ is:

$$
d(x,y)=\sqrt{\sum_{d=1}^{D}(x_d-y_d)^2}.
$$

## 1. Pairwise Distances and Classification

For one query, think of all training examples being ranked from closest to farthest:

```text
query q ── distances to all training points ── sort ── take first K
                                                       │
                                               class-label vote
                                                       │
                                                  prediction
```

`X[:, None, :]` adds a singleton axis to make shape `(N, 1, D)`. `Y[None, :, :]` has shape `(1, M, D)`. Broadcasting subtracts every pair to form `(N, M, D)` differences; summing over the feature axis returns the `(N, M)` distance matrix.

```python
import numpy as np


def pairwise_distances(X, Y):
    # X: (N, D), Y: (M, D)
    diff = X[:, None, :] - Y[None, :, :]  # (N, M, D)
    return np.sqrt(np.sum(diff ** 2, axis=-1))  # (N, M)


def knn_classify(X_train, y_train, X_test, K=3):
    M = X_test.shape[0]

    # Put queries first so each row contains distances to all training points.
    distances = pairwise_distances(X_test, X_train)  # (M, N)
    indices = np.argsort(distances, axis=1)[:, :K]    # (M, K)
    neighbors = y_train[indices]                      # (M, K)

    predictions = np.zeros(M, dtype=int)
    for i in range(M):
        counts = np.bincount(neighbors[i])
        predictions[i] = np.argmax(counts)

    return predictions
```

`argsort` returns training indices from smallest distance to largest. Indexing `y_train[indices]` retrieves the labels of those nearest examples. `bincount` counts each integer label, and `argmax` selects the majority class.

### Classification example

The source uses these training points:

```text
Point       Class       Point       Class
(1, 2)        0         (6, 5)        1
(2, 3)        0         (7, 7)        1
(3, 3)        0         (8, 6)        1
```

For query `(4, 4)` and `K=3`, the nearest-neighbor ranking is:

```text
distance 1.41 : (3, 3), class 0
distance 2.24 : (2, 3), class 0
distance 2.24 : (6, 5), class 1
                 └── 2 votes for class 0, 1 vote for class 1
prediction: 0
```

The full source queries `[(2,2), (7,6), (4,4)]` and predicts `[0, 1, 0]`.

## 2. KNN Regression

Regression uses the same nearest-neighbor search, but averages the neighbors' numeric targets instead of voting for a class:

$$
\hat{y}(x)=\frac{1}{K}\sum_{i\in\mathcal{N}_K(x)}y_i,
$$

where $\mathcal{N}_K(x)$ is the set of indices of the $K$ nearest training samples.

```python
def knn_regress(X_train, y_train, X_test, K=3):
    distances = pairwise_distances(X_test, X_train)
    indices = np.argsort(distances, axis=1)[:, :K]
    neighbors = y_train[indices]
    return np.mean(neighbors, axis=1)
```

In the source's one-dimensional example:

```text
feature x:  0       1       2           4       5       6
target y:   0      query   20          40     query    60
                  <--- K=2 --->              <--- K=2 --->
```

Query `1` uses targets `0` and `20`, so it predicts `10`. Query `5` uses `40` and `60`, so it predicts `50`. The source output is `[10., 50.]`.

## Practical and Numerical Notes

- **Feature scale matters.** Euclidean distance can be dominated by a feature with a much larger numeric range. Scale features using training-set statistics, then apply the same transformation to queries.
- **Class labels:** `np.bincount` expects nonnegative integer labels. If vote counts tie, `np.argmax` returns the smallest label index among the tied labels.
- **Choose `K` within the data size:** this implementation assumes `1 <= K <= N`.
- **Cost:** Pairwise distance calculation costs $O(MND)$ and temporarily stores an `(M, N, D)` difference array. Sorting each query's $N$ distances costs about $O(MN\log N)$.
- **Large values:** Squaring very large feature differences can overflow. Feature scaling also helps keep the distances in a manageable range.
