# Inverted Dropout

Dropout randomly removes activations during training so a model cannot rely on the same activations on every pass. Inverted dropout scales the activations that remain, preserving each activation's expected value and making inference an identity operation. This guide follows `inverted-dropout.py`.

## Notation

| Symbol | Shape | Meaning |
| :--- | :--- | :--- |
| $x$ | arbitrary | Input array of activations |
| $p$ | scalar | Probability of dropping an element, $0 \le p < 1$ |
| $q = 1-p$ | scalar | Probability of keeping an element |
| $m$ | same as $x$ | Random mask; each entry is 1 with probability $q$, otherwise 0 |
| $y$ | same as $x$ | Output array |

## Basic formulation

For each element, independently sample a mask and apply it during training:

$$
m_i \sim \operatorname{Bernoulli}(q), \qquad y_i = x_i\frac{m_i}{q}
$$

A dropped element becomes zero. A retained element becomes $x_i/q$. For example, if $p=0.5$, each element is dropped half the time and doubled otherwise. For a fixed input $x_i$, its average output over repeated random masks is therefore

$$
\mathbb{E}[y_i\mid x_i]
= p\cdot 0 + (1-p)\frac{x_i}{1-p}
= x_i.
$$

This is an expectation over masks, not a guarantee that every sampled array has the same mean as its input. Scaling during training is why it is called *inverted* dropout: at evaluation time, no mask or scaling is needed, so $y=x$.

### NumPy implementation

```python
import numpy as np


def dropout(x, p=0.5, training=True):
    if not 0 <= p < 1:
        raise ValueError("p must satisfy 0 <= p < 1")
    if not training or p == 0:
        return x

    mask = np.random.random(x.shape) >= p
    return x * mask / (1 - p)
```

`np.random.random(x.shape)` draws one value per input element. Comparing each draw with `p` creates a Boolean mask with the same shape; multiplication treats `False` as zero and `True` as one. Evaluation and `p=0` return the input unchanged.

### Example

```python
np.random.seed(42)
x = np.ones((4, 5))
print(dropout(x, p=0.5))
# [[0. 2. 2. 2. 0.]
#  [0. 0. 2. 2. 2.]
#  [0. 2. 2. 0. 0.]
#  [0. 0. 2. 0. 0.]]

print(dropout(np.array([1., 2.]), p=0.5, training=False))
# [1. 2.]
```

## Numerical behavior and limits

The same implementation is used here; there is no separate numerically stable version in the source. The restriction $p<1$ prevents division by zero. As $p$ approaches 1, however, the few retained values are multiplied by a large factor. For a fixed input, the output variance is

$$
\operatorname{Var}(y_i\mid x_i)=x_i^2\frac{p}{1-p}.
$$

Thus the expected value stays the same, but individual outputs become more variable and may overflow at extreme magnitudes. Epsilon or clipping is not part of this implementation.

A large sample illustrates the expectation, subject to sampling variation:

```python
np.random.seed(42)
out = dropout(np.ones(100_000), p=0.3)
print(out.mean())
# Approximately 1.0
```

In an MLP, dropout is commonly applied to hidden activations during training, often after a nonlinearity. Evaluation leaves activations intact to avoid random predictions.

## Follow-up questions

**Why scale by $1/(1-p)$?** An activation survives with probability $1-p$. Scaling the survivors by its reciprocal compensates for the dropped values, so the expected output equals the input, as derived above.

**Why use inverted dropout instead of standard dropout?** Standard dropout leaves surviving activations unscaled during training and scales them by the keep probability at inference. Inverted dropout does the scaling during training, so inference simply returns the input.

**Why apply dropout only during training?** Random masks encourage the model to learn without depending on the same activations every time. At inference, removing activations would introduce randomness into predictions; using all activations gives a deterministic dropout layer.

**What happens as $p$ approaches 1?** Fewer activations survive, and each survivor is scaled by a larger factor. The expected output remains unchanged, but its variance grows as shown above. At $p=1$, the scaling would divide by zero, so the function rejects it.

**Where would you apply dropout in an MLP?** Commonly to hidden activations during training, often after a nonlinear activation. The appropriate placement and rate depend on the model and task.
