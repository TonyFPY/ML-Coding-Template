

"""
K-Means

X: (N, D) input data
K: number of clusters

Returns:
    centroids: (K, D)
    labels: (N,)
"""
import numpy as np

def kmeans(X, K, max_iters=100, tol=1e-4):
    # X: (N, D)
    N, D = X.shape

    # 1. Initialize centroids
    indices = np.random.choice(N, K, replace=False)
    centroids = X[indices].astype(float).copy()  # (K, D)

    for _ in range(max_iters):
        # 2. Squared Euclidean distances: (N, K)
        distances = np.sum(
            (X[:, None, :] - centroids[None, :, :]) ** 2,
            axis=-1
        )

        # 3. Assign samples to nearest centroids: (N,)
        labels = np.argmin(distances, axis=1)

        # 4. Update centroids
        new_centroids = centroids.copy()

        for k in range(K):
            points = X[labels == k]

            if len(points) > 0:
                new_centroids[k] = np.mean(points, axis=0)

        # 5. Check convergence
        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids

        if shift < tol:
            break

    # 6. Recalculate labels using final centroids
    distances = np.sum(
        (X[:, None, :] - centroids[None, :, :]) ** 2,
        axis=-1
    )
    labels = np.argmin(distances, axis=1)

    return centroids, labels

        

    

np.random.seed(42)

X = np.array([
    [1., 1.],
    [1., 2.],
    [2., 1.],
    [8., 8.],
    [8., 9.],
    [9., 8.]
])

centroids, labels = kmeans(X, K=2)

print("Centroids:\n", centroids)
print("Labels:", labels)

X = np.array([
    [1., 2.],
    [3., 4.],
    [5., 6.]
])

centroids, labels = kmeans(X, K=1)

print(centroids)
print(labels)

X = np.array([
    [0., 0.],
    [0., 0.],
    [0., 0.],
    [5., 5.],
    [5., 5.]
])

centroids, labels = kmeans(X, K=3)

assert np.isfinite(centroids).all()
assert labels.shape == (5,)
assert centroids.shape == (3, 2)

'''
Interview follow-ups
Once you've implemented K-means, I'll ask you to explain:
1. Empty clusters: Why can a centroid receive zero assigned points? Compare keeping its previous position versus reinitializing it.
2. Complexity: Why is the standard per-iteration runtime \(O(NKD)\), and what is the memory trade-off between broadcasting and processing points individually?
3. Convergence: Why does K-means generally converge to a local optimum rather than necessarily finding the global minimum?
4. Initialization: What is the difference between random initialization and K-means++?
5. Distance choice: Why is squared Euclidean distance sufficient for assigning points?

'''