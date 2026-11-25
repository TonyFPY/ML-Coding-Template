# Scaled Dot-Product and Multi-Head Attention

Attention lets each query combine values from relevant keys. This guide introduces the calculation, gives basic and numerically stable NumPy versions, then presents a compact PyTorch implementation of single-head and multi-head self-attention.

## Notation

| Symbol | Shape | Meaning |
| :--- | :--- | :--- |
| $Q$ | $(B, T_Q, d_k)$ | Queries: what each position looks for |
| $K$ | $(B, T_K, d_k)$ | Keys: what each position offers for matching |
| $V$ | $(B, T_K, d_v)$ | Values: information to combine |
| $S$ | $(B, T_Q, T_K)$ | Scores between each query and key |
| $A$ | $(B, T_Q, T_K)$ | Attention weights over keys |
| $O$ | $(B, T_Q, d_v)$ | Weighted values for each query |

$B$ is batch size. In self-attention, queries, keys, and values come from the same input, so $T_Q=T_K=T$. A Boolean mask uses `True` for an allowed query-key pair.

## Version 1 — Basic Attention

For each query, compare it with every key, turn the scores into weights, and use those weights to average the values:

$$
S = \frac{QK^\top}{\sqrt{d_k}},\qquad
A_{ij} = \frac{e^{S_{ij}}}{\sum_{k=1}^{T_K} e^{S_{ik}}},\qquad
O = AV.
$$

The transpose swaps only the last two axes: $(B,T_K,d_k)$ becomes $(B,d_k,T_K)$. The denominator of softmax sums over keys for each query, so each row of $A$ sums to one. Scaling by $\sqrt{d_k}$ keeps dot products from growing too large as the key dimension increases.

For ordinary finite scores, a direct NumPy version is:

```python
import numpy as np


def basic_attention(Q, K, V, mask=None):
    scores = Q @ K.swapaxes(-1, -2) / np.sqrt(Q.shape[-1])
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    exp_scores = np.exp(scores)
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    return weights @ V, weights
```

A mask replaces disallowed scores with $-\infty$, giving them zero weight after softmax. For example, a lower-triangular causal mask allows each position to attend to itself and earlier positions only. Every query must have at least one allowed key.

## Version 2 — Numerically Stable Attention

Direct exponentiation can overflow for large positive scores. Subtracting each row's maximum leaves softmax unchanged while keeping its largest exponent at zero. The score calculation, masking, and weighted sum remain the same.

This is the NumPy implementation in `attention.py`:

```python
import numpy as np


def scaled_dot_product_attention(Q, K, V, mask=None):
    d = Q.shape[-1]
    scores = Q @ K.swapaxes(-1, -2) / np.sqrt(d)

    if mask is not None:
        scores = np.where(mask, scores, -np.inf)

    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(shifted)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    output = attn_weights @ V
    return output, attn_weights
```

For finite allowed scores, shifting avoids exponential overflow. It does not prevent overflow in the earlier dot product. A fully masked row has no finite maximum: subtracting $-\infty$ from $-\infty$ produces `NaN`. The implementation assumes at least one allowed key per query.

### Causal example

```python
T = 4
mask = np.tril(np.ones((T, T), dtype=bool))
q = np.ones((1, T, 2))
k = np.ones((1, T, 2))
v = np.arange(8, dtype=float).reshape(1, T, 2)

out, weights = scaled_dot_product_attention(q, k, v, mask)
print(weights)
# [[[1.         0.         0.         0.        ]
#   [0.5        0.5        0.         0.        ]
#   [0.33333333 0.33333333 0.33333333 0.        ]
#   [0.25       0.25       0.25       0.25      ]]]
print(out)
# [[[0. 1.]
#   [1. 2.]
#   [2. 3.]
#   [3. 4.]]]
```

All allowed scores are equal, so each output is the average of the values available up to that position. For a numerical check, scores such as `[1000, 1001]` would overflow under direct exponentiation; shifting them to `[-1, 0]` gives finite weights, approximately `[0.269, 0.731]`.

## Interview-Ready PyTorch Implementation

This compact implementation includes scaled dot-product attention, single-head self-attention, and multi-head self-attention. For input $X$ of shape $(B,T,D)$, each attention head works over the $T$ sequence positions. A Boolean mask uses `True` for allowed query-key pairs and broadcasts over batches and heads.

```python
import math
import torch
import torch.nn as nn


def SDPA(Q, K, V, mask=None):
    """Scaled dot-product attention.

    Q: (..., Tq, d_k)
    K: (..., Tk, d_k)
    V: (..., Tk, d_v)
    mask: Boolean; True means allowed, False means masked.

    Returns:
        out:  (..., Tq, d_v)
        attn: (..., Tq, Tk)
    """
    d = Q.shape[-1]
    scores = Q @ K.transpose(-1, -2) / math.sqrt(d)

    if mask is not None:
        mask = torch.broadcast_to(mask, scores.shape)
        scores = scores.masked_fill(~mask, -torch.inf)

        # Make fully masked rows finite before softmax, then zero their weights.
        all_masked = ~mask.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(all_masked, 0.0)
        attn = torch.softmax(scores, dim=-1).masked_fill(~mask, 0.0)
    else:
        attn = torch.softmax(scores, dim=-1)

    out = attn @ V
    return out, attn


class SingleHeadAttn(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, X, mask=None):
        # X: (B, T, d_model)
        Q = self.Wq(X)
        K = self.Wk(X)
        V = self.Wv(X)
        O, _ = SDPA(Q, K, V, mask)
        return self.Wo(O)


class MultiHeadAttn(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert num_heads > 0
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, X, mask=None):
        # X: (B, T, d_model)
        B, T, D = X.shape
        H, d = self.num_heads, self.d_head

        Q = self.Wq(X).reshape(B, T, H, d).transpose(1, 2)
        K = self.Wk(X).reshape(B, T, H, d).transpose(1, 2)
        V = self.Wv(X).reshape(B, T, H, d).transpose(1, 2)
        # Q, K, V: (B, H, T, d)

        if mask is not None and mask.ndim == 3:
            mask = mask[:, None, :, :]  # (B, T, T) -> (B, 1, T, T)

        O, _ = SDPA(Q, K, V, mask)  # O: (B, H, T, d)
        O = O.transpose(1, 2).reshape(B, T, D)  # concatenate heads
        return self.Wo(O)


# Shape check with a causal mask: each position attends to itself and earlier positions.
B, T, D, H = 2, 10, 128, 8
X = torch.randn(B, T, D)
mask = torch.ones(T, T, dtype=torch.bool, device=X.device).tril()

single = SingleHeadAttn(d_model=D)
assert single(X, mask).shape == (B, T, D)

multi = MultiHeadAttn(d_model=D, num_heads=H)
assert multi(X, mask).shape == (B, T, D)
```

The score tensor has shape `(B, H, T, T)`: for each query position, it stores one score per key position and head. Reshaping and transposing split `(B, T, D)` into `(B, H, T, d_head)`. The final transpose and reshape concatenate the heads back to `(B, T, D)`. A `(T, T)` mask broadcasts across batch and heads; a `(B, T, T)` mask gains a singleton head dimension.



## Interview Follow-ups

1. **Why must `d_model` be divisible by `num_heads`?** This implementation reshapes the features into equally sized heads, each of width `d_model // num_heads`.
2. **Why scale by $\sqrt{d_h}$, not $\sqrt{D}$?** Each head's dot product contains $d_h$ terms, so its score scale depends on the head width.
3. **Why separate the head dimension before attention?** Each head needs its own $T\times T$ score matrix and softmax distribution. A single dot product over all $D$ features would produce only one distribution.
4. **What does `Wo` do?** It learns how to mix the concatenated head outputs and returns a $D$-dimensional representation.
5. **What are the time and space costs?** For batch size $B$, sequence length $T$, model width $D$, and $H$ heads, projections cost $O(BTD^2)$ and attention costs $O(BT^2D)$ time. Materializing attention weights costs $O(BHT^2)$ space, alongside $O(BTD)$ activations and $O(D^2)$ parameters; training may retain additional intermediates.
6. **Causal mask versus padding mask?** A causal mask hides future keys. A padding mask hides padding keys so real tokens do not attend to them. They can be combined; this `SDPA` implementation assigns zero attention weights to a fully masked row.
7. **Can one head of width $D$ replace multiple heads without changing the structure?** No. One head has one attention distribution per query; multiple heads can attend to different positions independently before their outputs are mixed.
