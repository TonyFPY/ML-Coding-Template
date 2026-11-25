# Two-Layer MLP: Forward Pass, Backpropagation, and SGD

This guide follows the current `two-layer-mlp.py`. It covers a NumPy MLP with one ReLU hidden layer and a softmax classifier. The forward and backward equations are correct for valid class IDs and ordinary finite logits. The current softmax is shifted for stability, but its probability-based cross-entropy can still become infinite for extreme wrong predictions; the stable logits-based alternative is shown below.

## Notation

For a batch of $B$ examples, $D$ input features, $H$ hidden units, and $C$ classes:

| Variable | Shape | Meaning |
| :--- | :--- | :--- |
| $X$ | $(B,D)$ | Input features |
| $Y$ | $(B,)$ | Integer class IDs from $0$ to $C-1$ |
| $W_1$ | $(D,H)$ | Input-to-hidden weights |
| $b_1$ | $(H,)$ | Hidden-layer bias |
| $Z_1$ | $(B,H)$ | Hidden-layer pre-activations |
| $A_1$ | $(B,H)$ | ReLU activations |
| $W_2$ | $(H,C)$ | Hidden-to-output weights |
| $b_2$ | $(C,)$ | Output-layer bias |
| $Z_2$ | $(B,C)$ | Class logits |
| $P$ | $(B,C)$ | Softmax probabilities |

Each row is one example, and each target is one class ID.

---

## MLP-CE — Two-Layer Classification

### Version 1 — Basic Classifier

#### Forward pass and loss

The network applies an affine layer, ReLU, and a second affine layer:

$$
Z_1=XW_1+b_1,
\qquad
A_1=\operatorname{ReLU}(Z_1)=\max(0,Z_1),
\qquad
Z_2=A_1W_2+b_2
$$

Softmax converts each row of logits into probabilities:

$$
p_{ic}=\frac{e^{z_{ic}}}{\sum_{k=1}^{C}e^{z_{ik}}}
$$

For target class $y_i$, the loss for example $i$ is the negative log probability assigned to that class. The batch loss is the mean:

$$
\ell_i=-\log(p_{i,y_i}),
\qquad
L=-\frac{1}{B}\sum_{i=1}^{B}\log(p_{i,y_i})
$$

#### Backpropagation

Let $Q$ be the one-hot target matrix. For mean softmax cross-entropy, the gradient with respect to the logits simplifies to:

$$
dZ_2=\frac{P-Q}{B}
$$

The second layer's gradients are:

$$
dW_2=A_1^\top dZ_2,
\qquad
db_2=\sum_{i=1}^{B}(dZ_2)_i
$$

Propagate through the second affine layer and the ReLU activation:

$$
dA_1=dZ_2W_2^\top,
\qquad
dZ_1=dA_1\odot\mathbf{1}[Z_1>0]
$$

The first layer's gradients are:

$$
dW_1=X^\top dZ_1,
\qquad
db_1=\sum_{i=1}^{B}(dZ_1)_i
$$

An SGD step subtracts the learning rate $\eta$ times each gradient from its parameter:

$$
W_1\leftarrow W_1-\eta dW_1,\quad
b_1\leftarrow b_1-\eta db_1,\quad
W_2\leftarrow W_2-\eta dW_2,\quad
b_2\leftarrow b_2-\eta db_2
$$

#### NumPy implementation

```python
import numpy as np


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    # Subtract each row's maximum before exponentiating.
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def ce_loss(y, p):
    # Select each example's true-class probability and average its negative log.
    rows = np.arange(y.shape[0])
    return -np.mean(np.log(p[rows, y]))


def mlp_forward(x, w1, b1, w2, b2, y):
    z1 = x @ w1 + b1
    a1 = relu(z1)
    z2 = a1 @ w2 + b2

    p = softmax(z2)
    loss = ce_loss(y, p)
    cache = (x, y, w2, z1, a1, p)
    return loss, cache


def mlp_backward(cache):
    x, y, w2, z1, a1, p = cache
    B = x.shape[0]

    # Build one-hot targets, then use dL/dZ2 = (P - Q) / B.
    one_hot = np.zeros_like(p)
    one_hot[np.arange(B), y] = 1
    dl_dz2 = (p - one_hot) / B

    dl_dw2 = a1.T @ dl_dz2
    dl_db2 = np.sum(dl_dz2, axis=0)

    dl_da1 = dl_dz2 @ w2.T
    dl_dz1 = dl_da1 * (z1 > 0)

    dl_dw1 = x.T @ dl_dz1
    dl_db1 = np.sum(dl_dz1, axis=0)
    # Match sgd's expected order: dW1, db1, dW2, db2.
    return dl_dw1, dl_db1, dl_dw2, dl_db2


def sgd(w1, b1, w2, b2, grads, lr):
    dw1, db1, dw2, db2 = grads
    w1 -= lr * dw1
    b1 -= lr * db1
    w2 -= lr * dw2
    b2 -= lr * db2
    return w1, b1, w2, b2
```

`one_hot[np.arange(B), y] = 1` puts a one at each example's target-class position. Subtracting this matrix from `p` gives $P-Q$. The returned gradient order matches the order unpacked by `sgd`.

The source's `mlp_forward` returns `(loss, cache)`. Its `mlp_backward` returns `(dW1, db1, dW2, db2)`, and `sgd` updates the parameter arrays in place.

---

### Version 2 — Numerical Stability

#### Stable softmax

Direct exponentiation can overflow for large logits. Subtracting the maximum logit $m_i=\max_c z_{ic}$ from every class score in a row leaves softmax unchanged:

$$
p_{ic}=\frac{e^{z_{ic}-m_i}}{\sum_{k=1}^{C}e^{z_{ik}-m_i}}
$$

All exponent arguments are non-positive, so this avoids overflow. The current `softmax` already uses this method.

#### Stable cross-entropy from logits

The current `ce_loss` computes the loss from probabilities. If a very unlikely target probability underflows to zero, `-log(0)` becomes infinity. For example, logits `[1000, -1000]` with target class `1` have a finite mathematical loss of $2000$, while the probability-based calculation can return infinity.

Compute the loss directly from logits instead. With shifted logits, the per-example loss is:

$$
\ell_i=
\log\left(\sum_{c=1}^{C}e^{z_{ic}-m_i}\right)
-(z_{i,y_i}-m_i)
$$

```python
def ce_loss_with_logits(y, logits):
    B = y.shape[0]
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1))
    target_logits = shifted[np.arange(B), y]
    return np.mean(log_sum_exp - target_logits)
```

Use this in `mlp_forward` in place of `ce_loss(y, p)`:

```python
loss = ce_loss_with_logits(y, z2)
```

The backward gradient remains $dZ_2=(P-Q)/B$. The source example uses $B=2$, $D=2$, $H=3$, and $C=2$, so the logits have shape $(2,2)$ and the gradients have shapes $(2,3)$, $(3,)$, $(3,2)$, and $(2,)$ in the returned order.

### Class-Based NumPy Implementation

This version wraps the classifier in `TwoLayerNN`, like the regression example below. `forward` returns class probabilities. Call `loss(y)` and `backward(y)` after `forward`; the loss uses cached logits, and `backward` updates the parameters in place. Targets are integer class IDs with shape `(B,)`.

```python
import numpy as np


class TwoLayerNN:
    def __init__(self, input_dim, hidden_dim, num_classes, lr=0.01):
        self.lr = lr
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, num_classes) * 0.01
        self.b2 = np.zeros((1, num_classes))

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, logits):
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.logits = self.a1 @ self.W2 + self.b2
        self.probs = self.softmax(self.logits)
        return self.probs

    def loss(self, y):
        B = y.shape[0]
        shifted = self.logits - np.max(self.logits, axis=1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1))
        target_logits = shifted[np.arange(B), y]
        return np.mean(log_sum_exp - target_logits)

    def backward(self, y):
        B = self.X.shape[0]

        # Subtract 1 from each example's target-class probability.
        dz2 = self.probs.copy()
        dz2[np.arange(B), y] -= 1
        dz2 /= B

        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0)

        dW1 = self.X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
```

The indexed update `dz2[np.arange(B), y] -= 1` applies the one-hot target subtraction to one class entry in each row, producing the familiar gradient $dZ_2=(P-Q)/B$. The loss is computed from logits with log-sum-exp, so a very unlikely target does not require taking `log(0)`.

### Training Example

```python
np.random.seed(0)
X = np.array([[1., 2.],
              [3., 4.]])
y = np.array([0, 1])  # integer class IDs, shape (B,)

model = TwoLayerNN(input_dim=2, hidden_dim=3, num_classes=2, lr=0.1)

for _ in range(1000):
    model.forward(X)
    model.backward(y)

model.forward(X)
print(model.loss(y))
```

The classifier and regressor examples each define a standalone class named `TwoLayerNN`; use the version that matches the target type and loss in a given script.

---

## MLP-MSE — Two-Layer Regression

This variant uses the same two-layer architecture but predicts continuous values. Its output layer is linear, so it does not apply softmax. It trains with mean squared error (MSE) instead of categorical cross-entropy.

### Forward pass and loss

For $N$ examples and $O$ output values per example:

$$
Z_1=XW_1+b_1,
\qquad
A_1=\operatorname{ReLU}(Z_1),
\qquad
\hat{Y}=A_1W_2+b_2
$$

The code's `np.mean` averages across all $N\times O$ prediction elements:

$$
L=\frac{1}{NO}\sum_{i=1}^{N}\sum_{j=1}^{O}(\hat{y}_{ij}-y_{ij})^2
$$

Therefore, the output gradient is:

$$
dZ_2=\frac{2}{NO}(\hat{Y}-Y)
$$

For one output per example ($O=1$), this reduces to $2(\hat{Y}-Y)/N$.

### Backpropagation and update

The remaining gradients use the same chain rule as the classifier, starting from the MSE output gradient:

$$
dW_2=A_1^\top dZ_2,
\qquad
db_2=\sum_{i=1}^{N}(dZ_2)_i
$$

$$
dA_1=dZ_2W_2^\top,
\qquad
dZ_1=dA_1\odot\mathbf{1}[Z_1>0]
$$

$$
dW_1=X^\top dZ_1,
\qquad
db_1=\sum_{i=1}^{N}(dZ_1)_i
$$

Each parameter is updated with $\theta\leftarrow\theta-\eta d\theta$.

### Refined NumPy implementation

```python
import numpy as np


class TwoLayerNN:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.01):
        self.lr = lr
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros((1, output_dim))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_backward(self, dout, x):
        return dout * (x > 0)

    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        return self.a1 @ self.W2 + self.b2

    def loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, y_pred, y_true):
        # Both arrays have shape (N, output_dim); np.mean averages all N*output_dim values.
        N, output_dim = y_true.shape

        dz2 = (2.0 / (N * output_dim)) * (y_pred - y_true)

        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = self.relu_backward(da1, self.z1)

        dW1 = self.X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
```

The factor in `dz2` is `2 / (N * output_dim)` because `np.mean` averages over both examples and output values. The pasted `2 / N` expression is correct when `output_dim == 1`, as in the example below.

### Training example

Targets for scalar regression are kept two-dimensional with shape `(N, 1)`, matching the model output:

```python
np.random.seed(0)
X = np.random.randn(10, 3)
y = np.random.randn(10, 1)

model = TwoLayerNN(input_dim=3, hidden_dim=5, output_dim=1, lr=0.01)

for _ in range(1000):
    y_pred = model.forward(X)
    model.backward(y_pred, y)

print(model.loss(model.forward(X), y))
```

This MSE section is for regression with continuous targets. The earlier softmax cross-entropy section is for multiclass classification with integer class IDs.
