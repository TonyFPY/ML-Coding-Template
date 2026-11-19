# A Classification Training Step

A training step runs a batch through a model, measures its error, computes gradients, and updates the model's parameters. The source uses a PyTorch classifier, cross-entropy loss, and AdamW.

## Notation

| Name | Shape | Meaning |
| :--- | :--- | :--- |
| `x` | $(B, 16)$ | Batch of input features |
| `y` | $(B,)$ | Integer class IDs, each between 0 and 9 |
| `y_pred` | $(B, 10)$ | Raw class scores (logits) from the model |
| `loss` | scalar | Mean cross-entropy over the batch |

The example has $B=2$. Its targets are sampled from classes 0, 1, and 2, although the model has ten output classes.

## Basic Version — What the Training Step Computes

For logits $z_{ic}$ of example $i$ and class $c$, softmax gives the probability of each class. Cross-entropy measures the negative log probability of the target class $y_i$:

$$
p_{ic}=\frac{e^{z_{ic}}}{\sum_{j=1}^{C}e^{z_{ij}}},\qquad
L=-\frac{1}{B}\sum_{i=1}^{B}\log p_{i,y_i}.
$$

The mean is over examples; softmax normalizes over classes within each example. With ordinary logits, this NumPy example illustrates the calculation:

```python
import numpy as np

logits = np.array([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]])
targets = np.array([0, 2])
exp_logits = np.exp(logits)
probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
loss = -np.log(probs[np.arange(len(targets)), targets]).mean()
print(loss)  # 0.40760596...
```

`np.arange(len(targets))` supplies the row indices; paired with `targets`, it selects the target-class probability from each row. This small example uses three classes to make the arithmetic easy to see; the source model uses ten.

## Numerically Stable Version — The PyTorch Training Step

Direct exponentiation can overflow for large logits, and a very small target probability can round to zero before taking its logarithm. The source passes raw logits directly to `nn.CrossEntropyLoss()`, which uses a numerically stable log-softmax and negative log-likelihood calculation. Do not apply softmax before this loss.

The equivalent per-example loss subtracts the row maximum $m_i=\max_c z_{ic}$ before exponentiating:

$$
\ell_i=\log\left(\sum_c e^{z_{ic}-m_i}\right)-(z_{i,y_i}-m_i).
$$

For instance, logits `[1000, 999, 998]` have the same probabilities and target-class loss as `[0, -1, -2]`; the shifted form avoids exponentiating the large positive values.

This is the training step from `training.py`:

```python
def train_step(model, optimizer, criterion, x, y):
    model.train()
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    return loss
```

1. `model.train()` selects training behavior for layers such as dropout and batch normalization.
2. `optimizer.zero_grad()` clears gradients from the previous step.
3. `model(x)` produces logits. `criterion(y_pred, y)` computes the mean loss.
4. `loss.backward()` computes gradients of the loss with respect to trainable parameters.
5. `optimizer.step()` uses those gradients to update the parameters.

The source's complete example calls the step once:

```python
import torch
import torch.nn as nn

torch.manual_seed(42)
model = nn.Sequential(
    nn.Linear(16, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
x = torch.randn(2, 16)
y = torch.randint(0, 3, (2,))

loss = train_step(model, optimizer, criterion, x, y)
print(loss.item())
```

`y` contains integer class IDs, not one-hot vectors. `loss.item()` converts the scalar tensor to a Python number for display; the returned `loss` itself remains a tensor. One step is a demonstration, not a complete training run.

## Interview Follow-ups

1. **Why does cross-entropy accept logits rather than probabilities?** It computes log-softmax and the loss together in a numerically stable form. Passing softmax probabilities would make it apply softmax again and compute the wrong objective.
2. **What if `optimizer.zero_grad()` is omitted?** PyTorch accumulates gradients across backward calls, so an update uses the new gradients plus those left from earlier steps.
3. **What changes between `model.train()` and `model.eval()`?** They switch behavior of certain layers: dropout is active in training, while batch normalization uses batch statistics in training and running statistics in evaluation. They do not themselves enable or disable gradient tracking.
4. **What does `loss.backward()` compute?** Autograd applies the chain rule through the recorded forward operations and accumulates gradients in the `.grad` fields of trainable leaf tensors. It does not update the parameters.
5. **How do you debug `NaN` or non-decreasing loss?** Check inputs, targets, logits, loss, and gradients for non-finite values; verify target ranges and shapes; then inspect the learning rate and whether parameters actually update. For a loss that stays flat, try overfitting a small batch to isolate the issue.
