'''
Scaled dot-product attention with NumPy
Q (B, TQ, dK)
K (B, TK, dK)
V (B, TK, dV)
'''

import numpy as np 

def scaled_dot_product_attention(Q, K, V, mask=None):
    d = Q.shape[-1]

    # 1. Calculate scaled attention scores
    scores = Q @ K.swapaxes(-1, -2) / np.sqrt(d) # (B, TQ, TK)

    # 2. Apply optional mask (True = allowed)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)

    # 3. Stable softmax along last dimension
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(shifted)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # 4. Weighted sum of values
    output = attn_weights @ V # (B, TQ, dV)

    return output, attn_weights


if __name__ == "__main__":
    q = np.array([[[1., 0.],
                   [0., 1.]]])

    k = np.array([[[1., 0.],
                   [0., 1.]]])

    v = np.array([[[10., 0.],
                   [0., 20.]]])

    out, weights = scaled_dot_product_attention(q, k, v)

    print("Weights:\n", weights)
    print("Output:\n", out)

    T = 4
    mask = np.tril(np.ones((T, T), dtype=bool))

    q = np.ones((1, T, 2))
    k = np.ones((1, T, 2))
    v = np.arange(8, dtype=float).reshape(1, T, 2)

    out, weights = scaled_dot_product_attention(q, k, v, mask)

    print(weights)
    print(out)

'''
Single-head self-attention with PyTorch
'''

import torch
import torch.nn as nn
import math

class SingleHeadAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.Wk = nn.Linear(d, d)
        self.Wq = nn.Linear(d, d)
        self.Wv = nn.Linear(d, d)

    def forward(self, X, mask=None): # x: (B, T, D)
        K = self.Wk(X) # (B, T, d)
        Q = self.Wq(X) # (B, T, d)
        V = self.Wv(X) # (B, T, d)

        d = K.shape[-1]
        scores = Q @ K.transpose(-2, -1) / math.sqrt(d) # (B, T, T) # d ** 0.5

        if mask is not None:
            scores = scores.masked_fill(~mask, -torch.inf)

        # softmax
        attn = torch.softmax(scores, dim=-1)

        return attn @ V # (B, T, d)


if __name__ == "__main__":
    # Test 1: Output shape
    torch.manual_seed(42)

    X = torch.randn(2, 5, 8)
    model = SingleHeadAttention(d=8)

    out = model(X)
    print(out.shape)
    # torch.Size([2, 5, 8])

    # Test 2: Causal mask
    T = X.shape[1]
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))

    out = model(X, mask)
    print(out.shape)
    # torch.Size([2, 5, 8])


    # Test 3: Gradient propagation
    loss = out.sum()
    loss.backward()

    print(model.Wq.weight.grad is not None)
    print(model.Wk.weight.grad is not None)
    print(model.Wv.weight.grad is not None)
    # True
    # True
    # True

'''
Multi-head self-attention with PyTorch
'''

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = d_model // num_heads

        self.Wk = nn.Linear(d_model, d_model)
        self.Wq = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, X, mask=None): # X.shape (B, T, D)
        B, T, D = X.shape
        
        K = self.Wk(X) # (B, T, d_model)
        Q = self.Wq(X)
        V = self.Wv(X)

        K = K.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
        Q = Q.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
        V = V.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)

        if mask is not None:
            if mask.ndim == 3:  # (B, T, T) -> (B, 1, T, T)
                mask = mask[:, None, :, :]
            scores = scores.masked_fill(~mask, -torch.inf)

        attn = torch.softmax(scores, dim=-1) # (B, H, T, T)
        out = attn @ V # (B, H, T, d)

        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.d_head)

        return self.Wo(out)


if __name__ == "__main__":
    torch.manual_seed(42)

    x = torch.randn(2, 5, 8)
    model = MultiHeadAttention(d_model=8, num_heads=2)

    out = model(x)

    assert out.shape == (2, 5, 8)
    print("Test 1 passed")

    T = 5
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))

    out = model(x, mask=mask)

    assert out.shape == (2, 5, 8)
    assert torch.isfinite(out).all()
    print("Test 2 passed")

    model.zero_grad()

    out = model(x)
    loss = out.square().mean()
    loss.backward()

    for layer in [model.Wq, model.Wk, model.Wv, model.Wo]:
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()

    print("Test 3 passed")

    torch.manual_seed(42)

    x1 = torch.randn(1, 5, 8)
    x2 = x1.clone()

    # Change only the final token
    x2[:, -1, :] += 100.0

    model = MultiHeadAttention(d_model=8, num_heads=2)

    mask = torch.tril(torch.ones(5, 5, dtype=torch.bool))

    y1 = model(x1, mask)
    y2 = model(x2, mask)

    # Earlier outputs must remain unchanged
    assert torch.allclose(
        y1[:, :-1, :],
        y2[:, :-1, :],
        atol=1e-5
    )

    print("Test 4 passed")

'''
6. Interview follow-ups
After finishing the implementation, prepare to answer these questions without referring to notes:
1. Why must d_model be divisible by num_heads in this implementation?
2. Why do we scale attention scores by \(\sqrt{d_h}\), not \(\sqrt{D}\)?
3. Why must the head dimension be separated explicitly before attention?
4. What is the purpose of the final output projection Wo?
5. What are the time and space complexities of standard MHA, including linear projections?
6. What is the difference between a causal mask and a padding mask?
7. Can multi-head attention be replaced by a single attention head of dimension \(D\) without changing its representational structure?
'''