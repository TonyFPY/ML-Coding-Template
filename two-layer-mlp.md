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

## Version 1 — Basic Two-Layer MLP

### Forward pass and loss

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

### Backpropagation

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

### NumPy implementation

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

## Version 2 — Numerical Stability

### Stable softmax

Direct exponentiation can overflow for large logits. Subtracting the maximum logit $m_i=\max_c z_{ic}$ from every class score in a row leaves softmax unchanged:

$$
p_{ic}=\frac{e^{z_{ic}-m_i}}{\sum_{k=1}^{C}e^{z_{ik}-m_i}}
$$

All exponent arguments are non-positive, so this avoids overflow. The current `softmax` already uses this method.

### Stable cross-entropy from logits

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
