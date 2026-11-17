# Softmax Cross-Entropy: Basic and Numerically Stable Versions

This guide builds multiclass classification loss in two stages. The basic version introduces softmax, categorical cross-entropy, and the gradient with respect to logits. The second version computes softmax and cross-entropy in numerically stable forms.

## Notation

For a batch of $B$ examples and $C$ classes:

| Variable | Shape | Meaning |
| :--- | :--- | :--- |
| $Z$ | $(B, C)$ | Logits; row $z_i$ contains one example's class scores |
| $Y$ | $(B,)$ | Integer class targets in $\{0, \ldots, C-1\}$ |
| $P$ | $(B, C)$ | Softmax probabilities |
| $Q$ | $(B, C)$ | One-hot representation of the targets |

The loss is averaged over examples. Each example has exactly one target class.

---

## Version 1 — Basic Softmax Cross-Entropy

This version applies the standard formulas directly. Assume logits are in an ordinary range where exponentials and probabilities can be represented accurately.

### Softmax and cross-entropy

For example $i$ and class $c$, softmax converts logits into probabilities:

$$
p_{ic} = \frac{e^{z_{ic}}}{\sum_{j=1}^{C}e^{z_{ij}}}
$$

If the true class for example $i$ is $y_i$, its cross-entropy loss is the negative log probability assigned to that class:

$$
\ell_i = -\log(p_{i,y_i})
$$

The mean batch loss is $L = \frac{1}{B}\sum_{i=1}^B \ell_i$.

### Gradient derivation

Let $q_{ic}=1$ when $c=y_i$ and $0$ otherwise. Then the per-example loss can also be written as:

$$
\ell_i = -\sum_{c=1}^{C}q_{ic}\log(p_{ic})
$$

The softmax derivative is:

$$
\frac{\partial p_{ic}}{\partial z_{ik}}
= p_{ic}(\mathbf{1}[c=k]-p_{ik})
$$

Applying the chain rule gives the simple logit gradient:

$$
\frac{\partial \ell_i}{\partial z_{ik}}
= \sum_{c=1}^{C} -q_{ic}(\mathbf{1}[c=k]-p_{ik})
= p_{ik}-q_{ik}
$$

For the mean batch loss, the gradient is:

$$
\nabla_Z L = \frac{P-Q}{B}
$$

### Basic NumPy implementation

```python
import numpy as np


def softmax(logits):
    logits = np.asarray(logits, dtype=float)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def cross_entropy(probabilities, targets):
    probabilities = np.asarray(probabilities, dtype=float)
    targets = np.asarray(targets, dtype=int)
    B = probabilities.shape[0]

    true_probs = probabilities[np.arange(B), targets]
    return -np.mean(np.log(true_probs))


def softmax_cross_entropy_backward(logits, targets):
    logits = np.asarray(logits, dtype=float)
    targets = np.asarray(targets, dtype=int)
    B = logits.shape[0]

    probs = softmax(logits)
    grad = probs.copy()
    grad[np.arange(B), targets] -= 1  # subtract 1 at each row's true-class column
    return grad / B
```

`np.arange(B)` gives the row indices `[0, 1, ..., B - 1]`. Paired with `targets`, NumPy selects one element from each row: `grad[i, targets[i]]`. Subtracting `1` there turns each probability row into $p_i-q_i$, where $q_i$ is one-hot at the true class. Dividing by `B` then gives the gradient of the mean loss.

Example:

```python
logits = np.array([
    [2.0, 1.0, 0.0],
    [0.0, 1.0, 2.0],
])
targets = np.array([0, 2])

probs = softmax(logits)
print(probs)
# [[0.66524096 0.24472847 0.09003057]
#  [0.09003057 0.24472847 0.66524096]]
print(cross_entropy(probs, targets))
# 0.40760596...
```

---

## Version 2 — Numerically Stable Softmax Cross-Entropy

The formulas and gradient stay the same. The stable version shifts each row of logits before exponentiating and computes the loss directly from logits:

- Large logits can make $e^{z_{ic}}$ overflow.
- Softmax probabilities can round to zero for low-probability classes, making $\log(p_{i,y_i})$ infinite.

### Stable softmax

Softmax is unchanged if the same constant is subtracted from every class logit in a row. Let $m_i = \max_c z_{ic}$. Then:

$$
p_{ic} = \frac{e^{z_{ic}-m_i}}{\sum_{j=1}^{C}e^{z_{ij}-m_i}}
$$

Every exponent argument is non-positive, and at least one is zero, so the exponentials cannot overflow and the denominator is at least one.

### Cross-entropy directly from logits

For target class $y_i$, the loss can be written without first computing probabilities:

$$
\ell_i = \log\left(\sum_{c=1}^{C}e^{z_{ic}}\right)-z_{i,y_i}
$$

Subtract $m_i$ inside the log-sum-exp and cancel it against the shifted target logit:

$$
\ell_i
= \log\left(\sum_{c=1}^{C}e^{z_{ic}-m_i}\right) - (z_{i,y_i}-m_i)
$$

The sum uses the same bounded exponentials as stable softmax. The batch loss is the mean of these per-example losses.

### Stable NumPy implementation

```python
def stable_softmax(logits):
    logits = np.asarray(logits, dtype=float)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy_with_logits(logits, targets):
    logits = np.asarray(logits, dtype=float)
    targets = np.asarray(targets, dtype=int)
    B = logits.shape[0]

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1))
    true_logits = shifted[np.arange(B), targets]
    return np.mean(log_sum_exp - true_logits)


def softmax_cross_entropy_backward(logits, targets):
    logits = np.asarray(logits, dtype=float)
    targets = np.asarray(targets, dtype=int)
    B = logits.shape[0]

    probs = stable_softmax(logits)
    grad = probs.copy()
    grad[np.arange(B), targets] -= 1  # subtract 1 at each row's true-class column
    return grad / B
```

### Sanity checks

```python
# Equal logits give a uniform distribution.
logits = np.array([
    [1.0, 2.0, 3.0],
    [1.0, 1.0, 1.0],
])
print(stable_softmax(logits))
# [[0.09003057 0.24472847 0.66524096]
#  [0.33333333 0.33333333 0.33333333]]

# Large positive and negative offsets produce finite probabilities.
logits = np.array([
    [1000.0, 1001.0, 1002.0],
    [-1000.0, -1001.0, -1002.0],
])
print(stable_softmax(logits))
# [[0.09003057 0.24472847 0.66524096]
#  [0.66524096 0.24472847 0.09003057]]

# A highly confident wrong prediction has a finite loss.
logits = np.array([[1000.0, -1000.0]])
targets = np.array([1])
print(cross_entropy_with_logits(logits, targets))
# 2000.0

# The mean-loss gradient is (probabilities - one-hot targets) / batch size.
logits = np.array([
    [1.0, 2.0, 3.0],
    [1.0, 1.0, 1.0],
])
targets = np.array([2, 0])
print(softmax_cross_entropy_backward(logits, targets))
# [[ 0.04501529  0.12236424 -0.16737952]
#  [-0.33333333  0.16666667  0.16666667]]
```
