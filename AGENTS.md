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
