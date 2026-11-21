# K-Means: Math, NumPy Solution, and Interview Follow-ups

K-means groups points into $K$ clusters by alternating between assigning each point to its nearest centroid and moving each centroid to its cluster's mean. This guide follows [k-means.py](k-means.py), then adds a separately identified scaling variant for very large inputs.

## Notation

| Symbol / variable | Shape | Meaning |
| :--- | :--- | :--- |
| $X$ | $(N,D)$ | $N$ input points, each with $D$ features |
| $M$ / `centroids` | $(K,D)$ | One centroid $\mu_k$ per cluster |
| $z$ / `labels` | $(N,)$ | Cluster index $z_i\in\{0,\ldots,K-1\}$ for each point |
| $A$ / `distances` | $(N,K)$ | Squared distance from every point to every centroid |
| $C_k$ | Set of indices | Points currently assigned to cluster $k$ |
| `shift` | Scalar | Combined movement of all centroids |
| `tol` | Scalar | Stopping threshold for centroid movement |

Assume a finite, real NumPy array with $N,D>0$, an integer $1\le K\le N$, a positive integer `max_iters`, and positive `tol`. The source assumes valid inputs rather than checking these conditions.

## Version 1 — Basic K-Means

### Objective and assignment step

K-means minimizes the sum of squared distances to assigned centroids, also called inertia:

$$
J(M,z)=\sum_{i=1}^{N}\|x_i-\mu_{z_i}\|_2^2
=\sum_{i=1}^{N}\sum_{d=1}^{D}(x_{id}-\mu_{z_i,d})^2.
$$

Dividing $J$ by $ND$ gives a mean squared error over all input coordinates. This constant scaling changes the reported objective value but leaves its minimizers unchanged.

With centroids fixed, assign each point to its nearest centroid:

$$
A_{ik}=\sum_{d=1}^{D}(x_{id}-\mu_{kd})^2,
\qquad
z_i=\operatorname*{arg\,min}_{k\in\{0,\ldots,K-1\}} A_{ik}.
$$

```python
distances = np.sum(
    (X[:, None, :] - centroids[None, :, :]) ** 2,
    axis=-1,
)
labels = np.argmin(distances, axis=1)
```

`X[:, None, :]` has shape `(N, 1, D)` and `centroids[None, :, :]` has shape `(1, K, D)`. Broadcasting forms all point-centroid differences with shape `(N, K, D)`. Summing only the feature axis gives `(N, K)`; `argmin(axis=1)` then selects one cluster per point, giving `(N,)`. Equal distances select the first centroid index.

Using `np.mean(..., axis=-1)` would divide each distance by the same $D$, preserving the assignments. Omitting `axis` would reduce all three dimensions to one scalar and lose the point-centroid comparisons.

### Why is the centroid the mean?

With assignments fixed, cluster $k$ contributes:

$$
J_k(\mu_k)=\sum_{i\in C_k}\|x_i-\mu_k\|_2^2.
$$

Differentiate with respect to its centroid and set the result to zero:

$$
\nabla_{\mu_k}J_k
=2\sum_{i\in C_k}(\mu_k-x_i)
=2|C_k|\mu_k-2\sum_{i\in C_k}x_i=0.
$$

For a nonempty cluster:

$$
\mu_k=\frac{1}{|C_k|}\sum_{i\in C_k}x_i.
$$

```python
points = X[labels == k]              # (number of assigned points, D)
new_centroids[k] = points.mean(axis=0)  # (D,)
```

The boolean mask selects rows assigned to cluster `k`; `axis=0` averages those rows feature by feature. An empty cluster has no mean, so the source keeps its previous centroid.

### Convergence: a scalar norm versus MSE

The source measures the combined centroid movement using the Frobenius norm:

$$
\Delta=M_{\mathrm{new}}-M,
\qquad
\text{shift}=\|\Delta\|_F
=\sqrt{\sum_{k=0}^{K-1}\sum_{d=1}^{D}\Delta_{kd}^2}.
$$

`np.linalg.norm(new_centroids - centroids)` reduces the whole `(K, D)` matrix to one scalar. The source stores the new centroids, then stops when `shift < tol`.

For a concrete difference matrix:

```python
delta = np.array([[3., 4.], [0., 0.]])  # K=2, D=2
```

| Expression | Shape | Value |
| :--- | :--- | :--- |
| `np.mean(delta ** 2, axis=-1)` | `(K,)` | `[12.5, 0.]` |
| `np.mean(delta ** 2)` | Scalar | `6.25` |
| `np.linalg.norm(delta, axis=1)` | `(K,)` | `[5., 0.]` |
| `np.linalg.norm(delta)` | Scalar | `5.0` |

`axis=-1` removes only the feature dimension, leaving one result per centroid. This is why the earlier per-centroid MSE was an array. A condition such as `if per_centroid_mse <= tol` cannot choose a single truth value when $K>1$; it needs an explicit reduction such as `np.all(...)`.

A scalar centroid MSE is also a valid stopping metric, but its threshold has a different scale:

$$
\operatorname{MSE}_{\mathrm{centroids}}
=\frac{1}{KD}\sum_{k,d}\Delta_{kd}^2
=\frac{\text{shift}^2}{KD}.
$$

Thus `shift < tol` corresponds to `centroid_mse < tol**2 / (K * D)`. Reusing the same numerical threshold for both tests changes the stopping behavior. Centroid-movement MSE compares two centroid matrices; the clustering objective compares points with their assigned centroids.

### Complete NumPy solution

This has the same function signature, calculations, empty-cluster policy, and return values as the source. Distances are vectorized over points and clusters; the centroid update loops over the $K$ clusters.

```python
import numpy as np


def kmeans(X, K, max_iters=100, tol=1e-4):
    N, D = X.shape

    # 1. Pick K distinct input rows as initial centroids.
    indices = np.random.choice(N, K, replace=False)
    centroids = X[indices].astype(float).copy()  # (K, D)

    for _ in range(max_iters):
        # 2. Compute squared distances to every centroid.
        distances = np.sum(
            (X[:, None, :] - centroids[None, :, :]) ** 2,
            axis=-1,
        )  # (N, K)

        # 3. Assign each point to its nearest centroid.
        labels = np.argmin(distances, axis=1)  # (N,)

        # 4. Update nonempty clusters; keep empty centroids unchanged.
        new_centroids = centroids.copy()
        for k in range(K):
            points = X[labels == k]
            if len(points) > 0:
                new_centroids[k] = np.mean(points, axis=0)

        # 5. Store the new centroids before checking convergence.
        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < tol:
            break

    # 6. Return labels assigned using the final centroids.
    distances = np.sum(
        (X[:, None, :] - centroids[None, :, :]) ** 2,
        axis=-1,
    )
    labels = np.argmin(distances, axis=1)
    return centroids, labels
```

The final assignment matters because labels inside the loop were calculated before the centroid update. Recomputing them makes the returned labels match the returned centroids, including when `max_iters` is reached. Reaching that limit does not guarantee full convergence.

### Examples from the source

Two separated groups:

```python
np.random.seed(42)
X = np.array([
    [1., 1.],
    [1., 2.],
    [2., 1.],
    [8., 8.],
    [8., 9.],
    [9., 8.],
])
centroids, labels = kmeans(X, K=2)
print(centroids)
# [[1.33333333 1.33333333]
#  [8.33333333 8.33333333]]
print(labels)
# [0 0 0 1 1 1]
```

Cluster IDs are arbitrary: another initialization can swap `0` and `1` while finding the same partition.

With one cluster, its centroid is the mean of all points:

```python
X = np.array([[1., 2.], [3., 4.], [5., 6.]])
centroids, labels = kmeans(X, K=1)
print(centroids)
# [[3. 4.]]
print(labels)
# [0 0 0]
```

Duplicate points with more clusters than distinct locations exercise the empty-cluster policy:

```python
X = np.array([[0., 0.], [0., 0.], [0., 0.], [5., 5.], [5., 5.]])
centroids, labels = kmeans(X, K=3)
assert np.isfinite(centroids).all()
assert centroids.shape == (3, 2)
assert labels.shape == (5,)
```

Keeping old centroids avoids taking the mean of an empty slice. Returning $K$ centroids does not guarantee $K$ occupied clusters.

## Version 2 — Numerically Stable Scaling Variant

The source applies the basic formulas directly. At very large finite magnitudes, squared differences, centroid sums, and the norm calculation can overflow. Converting centroids to floating point does not prevent this.

For large inputs, divide all coordinates by one shared scale before running the same algorithm:

$$
s=\max\left(1,\max_{i,d}|X_{id}|\right),
\qquad X'=X/s,\qquad M'=M/s.
$$

The normalized coordinates have magnitude at most one. In exact arithmetic:

$$
A'_{ik}=A_{ik}/s^2,
\qquad J'=J/s^2,
\qquad \text{shift}'=\text{shift}/s.
$$

The assignment and mean-update derivations are unchanged. To preserve the source's absolute movement threshold, use `tol / s` while clustering scaled data, then multiply the final centroids by $s$.

This wrapper is an additional variant for the guide; it is not present in `k-means.py`:

```python
def kmeans_scaled(X, K, max_iters=100, tol=1e-4):
    X = np.asarray(X, dtype=float)
    scale = max(1.0, np.max(np.abs(X)))
    centroids, labels = kmeans(
        X / scale,
        K,
        max_iters=max_iters,
        tol=tol / scale,
    )
    return centroids * scale, labels
```

This protects against overflow from a large overall input scale. Inputs must still be finite, and scaling cannot recover distinctions already lost to floating-point rounding. Extremely small differences can still underflow. Using a different scale for each feature changes the distance geometry and can change the clusters.

### Large-value example

The direct square overflows, while clustering the scaled data remains finite:

```python
X_large = np.array([[1e200], [2e200], [8e200], [9e200]])
with np.errstate(over="ignore"):
    raw_squared = (X_large - X_large[0]) ** 2
print(np.isinf(raw_squared).any())
# True

np.random.seed(42)
centroids, labels = kmeans_scaled(X_large, K=2, tol=1e190)
assert np.isfinite(centroids).all()
print(centroids / 1e200)
# [[1.5]
#  [8.5]]
print(labels)
# [0 0 1 1]
```

Here `tol=1e190` is expressed in the original coordinate units. The wrapper converts it to the corresponding threshold in scaled units.

## Interview Follow-up Answers

### 1. Why do empty clusters occur? Keep or reinitialize them?

A centroid can lose all its points when other centroids become closer. Duplicate initial coordinates can also cause ties; `np.argmin` assigns tied points to the first index, leaving other centroids empty. Sampling distinct rows does not guarantee distinct coordinates.

The source starts with `new_centroids = centroids.copy()` and updates only nonempty clusters. Keeping the old position is simple, stays finite, and may let that centroid receive points later, but it may remain unused. An empty cluster contributes nothing to the objective for the current assignments.

Reinitializing an empty centroid to a data point gives it another chance to attract points; choosing a poorly represented point is often more useful than choosing a point already near another centroid. This changes the source's policy and is not implemented here. When there are fewer distinct locations than $K$, the tie rule can still leave clusters empty.

### 2. Why is runtime $O(NKD)$, and what is the memory trade-off?

There are $NK$ point-centroid pairs, and each distance uses $D$ coordinates, so distance computation costs $O(NKD)$. Choosing labels costs $O(NK)$. The source's centroid loop also scans the $N$ labels for each of $K$ clusters, then averages assigned points; the full iteration remains $O(NKD)$.

For $T$ iterations, runtime is $O(TNKD)$, plus an $O(NKD)$ final assignment. Broadcasting materializes an `(N, K, D)` temporary and an `(N, K)` distance matrix, so its extra memory is dominated by $O(NKD)$.

Processing one point at a time needs only an `(K, D)` distance temporary, centroid sums/counts, and the returned labels. It keeps the same asymptotic runtime but introduces Python-loop overhead. To preserve this algorithm, keep centroids fixed while accumulating a full iteration's assignments, then update their means together. Processing chunks offers a middle ground for memory and speed.

### 3. Why does K-means generally find a local solution?

Each assignment step minimizes the objective with centroids fixed. Each nonempty centroid update minimizes it with assignments fixed. Consequently, the objective cannot increase in exact arithmetic and is bounded below by zero.

However, jointly choosing assignments and centroids is a nonconvex problem. The algorithm improves one partition at a time and can settle into a partition that is worse than one reached from another initialization. Neither a small centroid movement nor the iteration limit proves global optimality. Multiple random starts can reduce sensitivity by selecting the run with the smallest final inertia.

### 4. Random initialization versus K-means++?

The source picks $K$ distinct input row indices uniformly with `np.random.choice(N, K, replace=False)`. Several selected points can come from the same natural group, and distinct rows can have identical values.

K-means++ first selects one point uniformly. For each later centroid, let $a_i$ be the squared distance from point $i$ to its nearest already selected centroid:

$$
a_i=\min_{\mu\ \mathrm{selected}}\|x_i-\mu\|_2^2,
\qquad
\Pr(\text{select }x_i)=\frac{a_i}{\sum_j a_j}.
$$

This favors points far from existing centroids and usually gives a better starting spread. It does not guarantee the global optimum. If every $a_i=0$, distance-based sampling has no positive weights; an implementation needs a fallback, such as choosing a remaining row. K-means++ is not implemented in the source.

### 5. Why are squared Euclidean distances sufficient?

Square root is strictly increasing on nonnegative values, so it cannot change the nearest centroid:

$$
\operatorname*{arg\,min}_k\|x_i-\mu_k\|_2
=\operatorname*{arg\,min}_k\|x_i-\mu_k\|_2^2.
$$

Skipping the square root saves work during assignment. The squared-distance objective also makes the arithmetic mean the optimal centroid, as derived above. Replacing that objective with a sum of unsquared distances would generally require a different centroid update.
