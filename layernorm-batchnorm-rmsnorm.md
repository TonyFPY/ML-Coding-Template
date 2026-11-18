# LayerNorm, BatchNorm, and RMSNorm

This guide follows `layernorm-batchnorm-rmsnorm.py`. It explains the normalization axes, affine parameters, BatchNorm running statistics, and the role of epsilon. The formulas are correct for the file's two-dimensional inputs; a BatchNorm running-statistics shape detail is noted in the stability section.

## Notation

Let $X\in\mathbb{R}^{B\times D}$ be a batch with $B$ examples and $D$ features.

| Variable | Shape | Meaning |
| :--- | :--- | :--- |
| $X$ | $(B,D)$ | Input batch |
| $\gamma$ | $(D,)$ | Learned scale, broadcast over examples |
| $\beta$ | $(D,)$ | Learned shift, broadcast over examples |
| $\epsilon$ | scalar | Small positive constant added to the denominator calculation |
| $\mu_{run}$ | $(D,)$ or $(1,D)$ | BatchNorm running mean |
| $\sigma^2_{run}$ | $(D,)$ or $(1,D)$ | BatchNorm running variance |

---

## Version 1 — Basic Normalization Formulas

### LayerNorm

LayerNorm computes the mean and variance across the features of each example. For row $i$:

$$
\mu_i=\frac{1}{D}\sum_{j=1}^{D}x_{ij},
\qquad
\sigma_i^2=\frac{1}{D}\sum_{j=1}^{D}(x_{ij}-\mu_i)^2
$$

It normalizes that row, then applies a learned scale and shift:

$$
\hat{x}_{ij}=\frac{x_{ij}-\mu_i}{\sqrt{\sigma_i^2+\epsilon}},
\qquad
y_{ij}=\gamma_j\hat{x}_{ij}+\beta_j
$$

```python
def layer_norm(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return x_norm * gamma + beta
```

### BatchNorm

During training, BatchNorm computes a mean and variance for each feature across the batch:

$$
\mu_j=\frac{1}{B}\sum_{i=1}^{B}x_{ij},
\qquad
\sigma_j^2=\frac{1}{B}\sum_{i=1}^{B}(x_{ij}-\mu_j)^2
$$

It normalizes each feature, then applies scale and shift:

$$
\hat{x}_{ij}=\frac{x_{ij}-\mu_j}{\sqrt{\sigma_j^2+\epsilon}},
\qquad
y_{ij}=\gamma_j\hat{x}_{ij}+\beta_j
$$

The code updates its running statistics using momentum $m$:

$$
\mu_{run}\leftarrow m\mu_{batch}+(1-m)\mu_{run},
\qquad
\sigma^2_{run}\leftarrow m\sigma^2_{batch}+(1-m)\sigma^2_{run}
$$

At inference time, it uses those running values instead of statistics from the current input batch:

```python
def batch_norm(
    x, gamma, beta, running_mean, running_var,
    training=True, momentum=0.1, eps=1e-5
):
    if training:
        mean = np.mean(x, axis=0, keepdims=True)
        var = np.var(x, axis=0, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)

        running_mean = momentum * mean + (1 - momentum) * running_mean
        running_var = momentum * var + (1 - momentum) * running_var
    else:
        x_norm = (x - running_mean) / np.sqrt(running_var + eps)

    output = x_norm * gamma + beta
    return output, running_mean, running_var
```

### RMSNorm

RMSNorm scales each example by its root-mean-square value across features. It does not subtract the mean and this implementation has no learned shift parameter:

$$
r_i=\sqrt{\frac{1}{D}\sum_{j=1}^{D}x_{ij}^2+\epsilon},
\qquad
y_{ij}=\gamma_j\frac{x_{ij}}{r_i}
$$

```python
def rms_norm(x, gamma, eps=1e-5):
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    x_norm = x / rms
    return x_norm * gamma
```

### Which axes are normalized?

| Method | Statistics are computed over | Per-example centering? | Learned shift? |
| :--- | :--- | :--- | :--- |
| LayerNorm | Features in each row (`axis=-1`) | Yes | Yes, $\beta$ |
| BatchNorm in training | Examples in each feature (`axis=0`) | Yes | Yes, $\beta$ |
| RMSNorm | Squared features in each row (`axis=-1`) | No | No in this implementation |

---

## Version 2 — Epsilon and Numerical Notes

Each denominator adds $\epsilon$ before taking the square root. This prevents division by zero when the measured variance or mean square is zero. For a constant row, LayerNorm produces its shift $\beta$; for a zero row, RMSNorm produces zero.

Epsilon does not protect every computation from overflow. Squared deviations in `np.var` and `x ** 2` in RMSNorm can overflow for extremely large values. The implementation is intended for ordinary finite model activations.

### BatchNorm running-statistics shape

The example initializes `running_mean` and `running_var` with shape `(D,)`. In training, `mean` and `var` have shape `(1, D)` because `keepdims=True`; combining them with the running vectors returns statistics shaped `(1, D)`. Inference still works through broadcasting, but the returned shape changes. Keep the state shape consistent by either initializing it as `(1, D)` or squeezing the batch statistics before updating one-dimensional state.

### Example behavior

With `gamma = np.ones(D)` and `beta = np.zeros(D)`:

- LayerNorm outputs each row with mean near zero and variance slightly below one because of $\epsilon$.
- Training BatchNorm outputs each feature with batch mean near zero and variance slightly below one.
- RMSNorm outputs each nonzero row with RMS slightly below one; it preserves the row's direction rather than centering it.

The file's examples also pass the BatchNorm running statistics from a training call into an inference call. Both `(D,)` and `(1,D)` state shapes broadcast with an input of shape `(B,D)`.
