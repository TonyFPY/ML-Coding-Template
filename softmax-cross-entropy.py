'''
Part A — Implement softmax and cross-entropy

'''
import numpy as np

def stable_softmax(logits):
    # logits: (B, C)
    # return: (B, C)

    logits = np.asarray(logits, dtype=float)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy_with_logits(logits, targets):
    # logits: (B, C)
    # targets: (B,)
    # return: scalar
    logits = np.asarray(logits, dtype=float)
    targets = np.asarray(targets, dtype=int)

    B = logits.shape[0]

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1))

    true_logits = shifted[np.arange(B), targets]
    return np.mean(log_sum_exp - true_logits)

logits = np.array([
    [1., 2., 3.],
    [1., 1., 1.]
])
print(stable_softmax(logits))

logits = np.array([
    [1000., 1001., 1002.],
    [-1000., -1001., -1002.]
])
print(stable_softmax(logits))

logits = np.array([
    [2., 1., 0.],
    [0., 1., 2.]
])
targets = np.array([0, 2])
print(cross_entropy_with_logits(logits, targets))

logits = np.array([[1000., -1000.]])
targets = np.array([1])
print(cross_entropy_with_logits(logits, targets))

'''
Q2 Part B — Softmax Cross-Entropy Backward Pass
'''

def softmax_cross_entropy_backward(logits, targets):
    # logits: (B, C)
    # targets: (B,)

    B = logits.shape[0]

    # Step 1: Stable softmax
    probs = stable_softmax(logits)

    # Step 2: Subtract one-hot targets
    grad = probs.copy()
    grad[np.arange(B), targets] -= 1

    # Step 3: Mean loss gradient
    grad = grad / B

    return grad

logits = np.array([
    [1., 2., 3.],
    [1., 1., 1.]
])

targets = np.array([2, 0])

grad = softmax_cross_entropy_backward(logits, targets)

print(grad)