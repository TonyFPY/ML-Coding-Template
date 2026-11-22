'''

Implement a K-Nearest Neighbors (KNN) classifier from scratch using NumPy.

Given a training dataset, corresponding class labels and a set of query samples, predict the class of each query sample by majority voting among its \(k\) nearest training samples.

Use Euclidean distance:
\[
d(\mathbf{x},\mathbf{y})=
\sqrt{\sum_{i=1}^{D}(x_i-y_i)^2}
\]
'''


import numpy as np


import numpy as np

def pairwise_distances(X, Y):
    # X: (N, D)
    # Y: (M, D)

    diff = X[:, None, :] - Y[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))  # (N, M)

def knn_classify(X_train, y_train, X_test, K=3):
    M = X_test.shape[0]

    # Pairwise squared distances: (M, N)
    distances = pairwise_distances(X_test, X_train)

    # K nearest training samples: (M, K)
    indices = np.argsort(distances, axis=1)[:, :K]

    # Retrieve corresponding labels: (M, K)
    neighbors = y_train[indices]

    # Majority voting
    predictions = np.zeros(M, dtype=int)

    for i in range(M):
        counts = np.bincount(neighbors[i])
        predictions[i] = np.argmax(counts)

    return predictions


# Test
X_train = np.array([
    [1, 2],
    [2, 3],
    [3, 3],
    [6, 5],
    [7, 7],
    [8, 6]
])

y_train = np.array([0, 0, 0, 1, 1, 1])

X_test = np.array([
    [2, 2],
    [7, 6],
    [4, 4]
])

predictions = knn_classify(X_train, y_train, X_test, K=3)

print(predictions)
# Expected: [0 1 0]



def knn_regress(X_train, y_train, X_test, K=3):
    distances = pairwise_distances(X_test, X_train)

    indices = np.argsort(distances, axis=1)[:, :K]

    neighbors = y_train[indices]

    return np.mean(neighbors, axis=1)

X_train = np.array([[0.], [2.], [4.], [6.]])
y_train = np.array([0., 20., 40., 60.])

X_test = np.array([[1.], [5.]])

print(knn_regress(X_train, y_train, X_test, K=2))