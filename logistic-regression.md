# Logistic Regression: Basic and Numerically Stable Versions

This guide builds a binary logistic regression classifier in two stages. The basic version introduces the model, binary cross-entropy (BCE), gradients, and one SGD update. The second version replaces the probability-based loss and ordinary sigmoid with numerically stable implementations.

## Notation

For a batch of $B$ examples with $D$ features:

| Variable | Shape | Meaning |
| :--- | :--- | :--- |
| $X$ | $(B, D)$ | Feature matrix |
| $y$ | $(B,)$ | Binary targets in $\{0, 1\}$ |
| $w$ | $(D,)$ | Weight vector |
| $b$ | scalar | Bias |
| $z = Xw + b$ | $(B,)$ | Logits |
| $p = \sigma(z)$ | $(B,)$ | Predicted probabilities |

The mean loss is $L = \frac{1}{B}\sum_{i=1}^B \ell_i$.

---

## Version 1 — Basic Logistic Regression

This version uses the standard sigmoid and computes BCE from probabilities. Assume the logits are in an ordinary range where these direct formulas work as expected.

### Model and loss

The sigmoid maps a logit to a probability:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

For one target $y \in \{0,1\}$ and predicted probability $p=\sigma(z)$, binary cross-entropy is:

$$
\ell(p,y) = -\left[y\log(p) + (1-y)\log(1-p)\right]
$$

The batch loss is the mean of the per-example losses.

### Gradient derivation

Using $\frac{dp}{dz}=p(1-p)$, differentiate the loss first with respect to $p$:

$$
\frac{\partial \ell}{\partial p} = -\frac{y}{p} + \frac{1-y}{1-p}
$$

Then apply the chain rule:

$$
\frac{\partial \ell}{\partial z}
= \frac{\partial \ell}{\partial p}\frac{dp}{dz}
= \left(-\frac{y}{p} + \frac{1-y}{1-p}\right)p(1-p)
= p-y
$$

For the mean batch loss, define $dz = \nabla_z L = \frac{p-y}{B}$. Since $z=Xw+b$:

$$
\nabla_w L = X^\top dz = \frac{1}{B}X^\top(p-y),
\qquad
\frac{\partial L}{\partial b} = \sum_{i=1}^B dz_i = \frac{1}{B}\sum_{i=1}^B(p_i-y_i)
$$

An SGD update with learning rate $\eta$ is:

$$
w \leftarrow w - \eta\nabla_w L,
\qquad
b \leftarrow b - \eta\frac{\partial L}{\partial b}
$$

### Basic NumPy implementation

```python
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def binary_cross_entropy(p, y):
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def logistic_regression_step(X, y, w, b, lr):
    B = X.shape[0]

    z = X @ w + b
    p = sigmoid(z)
    loss = binary_cross_entropy(p, y)

    dz = (p - y) / B
    dw = X.T @ dz
    db = np.sum(dz)

    w = w - lr * dw
    b = b - lr * db
    return w, b, loss
```

Example:

```python
X = np.array([
    [ 1.0,  2.0],
    [ 2.0,  1.0],
    [-1.0, -2.0],
    [-2.0, -1.0],
])
y = np.array([1.0, 1.0, 0.0, 0.0])
w = np.zeros(2)
b = 0.0

w, b, loss = logistic_regression_step(X, y, w, b, lr=0.1)
print(w, b, loss)
# Weights: [0.075 0.075]; bias: 0.0; loss before update: 0.693147...
```

---

## Version 2 — Numerically Stable Logistic Regression

The model and gradients stay the same. The changes are to the sigmoid and loss calculations:

- In the ordinary sigmoid, $e^{-z}$ can overflow for a large negative $z$.
- Computing BCE after sigmoid can lose information when $p$ rounds to exactly $0$ or $1$, making a logarithm undefined.

### Stable sigmoid

For negative $z$, rewrite the sigmoid by multiplying numerator and denominator by $e^z$:

$$
\sigma(z) =
\begin{cases}
\frac{1}{1+e^{-z}} & z \ge 0 \\
\frac{e^z}{1+e^z} & z < 0
\end{cases}
$$

Each branch exponentiates a non-positive value, avoiding overflow.

### Stable BCE from logits

Starting with the probability-based BCE and substituting $p=\sigma(z)$ gives the equivalent logit form:

$$
\ell(z,y) = \log(1+e^z) - yz
$$

To avoid directly computing a potentially large exponential, use the identity:

$$
\ell(z,y) = \max(z,0) - yz + \log\left(1+e^{-|z|}\right)
$$

Because $-|z| \le 0$, the exponential is bounded between zero and one. `np.log1p` accurately evaluates $\log(1+x)$ when $x$ is small.

### Stable NumPy implementation

```python
def stable_sigmoid(z):
    z = np.asarray(z, dtype=float)
    p = np.empty_like(z)

    nonnegative = z >= 0
    p[nonnegative] = 1 / (1 + np.exp(-z[nonnegative]))

    negative = ~nonnegative
    ez = np.exp(z[negative])
    p[negative] = ez / (1 + ez)
    return p


def binary_cross_entropy_with_logits(z, y):
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    per_example_loss = (
        np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))
    )
    return np.mean(per_example_loss)


def stable_logistic_regression_step(X, y, w, b, lr):
    B = X.shape[0]

    z = X @ w + b
    p = stable_sigmoid(z)
    loss = binary_cross_entropy_with_logits(z, y)

    # Same gradient as the basic version.
    dz = (p - y) / B
    dw = X.T @ dz
    db = np.sum(dz)

    w = w - lr * dw
    b = b - lr * db
    return w, b, loss
```

### Sanity checks

```python
# Ordinary values agree with the expected sigmoid probabilities.
z = np.array([0.0, 2.0, -2.0])
print(stable_sigmoid(z))
# [0.5        0.88079708 0.11920292]

# Confident, correct predictions have zero loss at these magnitudes.
z = np.array([1000.0, -1000.0])
y = np.array([1.0, 0.0])
print(stable_sigmoid(z))
# [1. 0.]
print(binary_cross_entropy_with_logits(z, y))
# 0.0

# Confident, incorrect predictions retain a finite penalty.
y = np.array([0.0, 1.0])
print(binary_cross_entropy_with_logits(z, y))
# 1000.0
```
