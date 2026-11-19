# Markdown Guide Rules

These rules apply when creating or editing the Markdown guides in this repository.

## Shared rules

- Read the paired Python source before changing its guide. Treat the source as authoritative for function names, input shapes, constraints, calculations, and examples.
- Structure guides in this order: title and short overview, notation, a basic version, then a numerically stable version.
- Explain the basic formulation before discussing numerical failure modes. In the stable section, show what changes and reuse shared derivations instead of repeating them.
- Keep equations, gradient scaling, array shapes, code, and displayed outputs consistent with one another and with the source.
- Use NumPy and vectorized operations in examples. Preserve source constraints and avoid introducing unrelated APIs, training features, or optimizers.
- Keep explanations concise and interview-focused. Explain non-obvious indexing or broadcasting directly where it appears.
- Include examples that illustrate ordinary inputs and the numerical behavior the stable implementation is intended to handle.

## `logistic-regression.md`

Paired source: `logistic-regression.py`.

- Describe binary classification with logits $z=Xw+b$, binary targets, sigmoid probabilities, and binary cross-entropy.
- In the basic version, show the ordinary sigmoid and BCE from probabilities. In the stable version, show the piecewise sigmoid and BCE-with-logits formula.
- Keep the mean-loss gradients as $dz=(p-y)/B$, $dw=X^\top dz$, and $db=\sum_i dz_i$, followed by the SGD update present in the source.
- Preserve the source constraints: NumPy only, vectorized operations, valid matching targets, and no probability clipping as a replacement for a stable logits-based loss.

## `softmax-cross-entropy.md`

Paired source: `softmax-cross-entropy.py`.

- Describe multiclass, single-label classification with logits of shape $(B,C)$ and integer targets of shape $(B,)$.
- Normalize softmax across classes for each example (`axis=1`). In the basic version, show direct softmax and probability-based categorical cross-entropy. In the stable version, subtract each row's maximum and compute cross-entropy from shifted logits and the target-class logit.
- Derive and implement the mean-loss gradient $(P-Q)/B$. Explain that `grad[np.arange(B), targets] -= 1` selects each row's target-class entry and subtracts its one-hot target value.
- Keep examples consistent with the source's batch-major shapes and class-index targets. Do not describe this as multilabel classification or binary cross-entropy.

## `layernorm-batchnorm-rmsnorm.md`

Paired source: `layernorm-batchnorm-rmsnorm.py`.

- Describe input and affine parameter shapes for $(B,D)$ inputs. Make the statistic axes explicit: LayerNorm and RMSNorm normalize each row over features; BatchNorm normalizes each feature across the batch during training.
- Preserve each method's formula: LayerNorm centers and scales with variance; BatchNorm uses batch statistics in training and running statistics in inference; RMSNorm divides by root mean square without centering and has no learned shift in this implementation.
- Explain that epsilon is added inside the square root, and that epsilon avoids a zero denominator but does not prevent overflow at extreme magnitudes.
- Document BatchNorm's running-statistics update and the current `(D,)` to `(1,D)` shape expansion caused by `keepdims=True`; mention that inference broadcasts either shape.

## `two-layer-mlp.md`

Paired source: `two-layer-mlp.py`; the guide may also include a separately identified MSE regression variant.

- Keep the softmax cross-entropy classifier and the MSE regressor distinct: classification uses integer class IDs and softmax; regression uses continuous targets and a linear output.
- For the MSE defined as `np.mean((y_pred - y_true) ** 2)`, use output gradient `2 * (y_pred - y_true) / (N * output_dim)`. Note that this is `2 / N` only for a single output.
- Keep target and prediction shapes aligned, and document the current classifier's forward return, backward gradient order, and SGD argument order.
