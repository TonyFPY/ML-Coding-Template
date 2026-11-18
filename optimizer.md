# Optimizers: Momentum, Adam, and AdamW

Optimizers use gradients to update parameters. Momentum carries information from earlier steps; Adam also adjusts the step size separately for each parameter. AdamW adds decoupled weight decay. This guide follows `optimizer.py` and uses elementwise operations throughout.

## Notation

| Symbol | Shape | Meaning |
| :--- | :--- | :--- |
| $\theta$ | $(D,)$ | Parameters |
| $g_t$ | $(D,)$ | Gradient at step $t$ |
| $u_t$ | $(D,)$ | Momentum buffer |
| $m_t, v_t$ | $(D,)$ | Adam's first and second moment estimates |
| $\eta$ | scalar | Learning rate |
| $\mu, \beta_1, \beta_2$ | scalars | Momentum and moment decay coefficients |
| $\epsilon$ | scalar | Small positive denominator offset |
| $\lambda$ | scalar | AdamW weight-decay coefficient |

Adam's step counter starts at $t=1$; initialize its moment vectors and the momentum buffer to zero and carry the returned state into the next step.

## Basic version: SGD with momentum

Ordinary SGD moves opposite the current gradient: $\theta_t=\theta_{t-1}-\eta g_t$. Momentum adds a fraction of the previous buffer before taking the step:

$$
u_t=\mu u_{t-1}+g_t, \qquad
\theta_t=\theta_{t-1}-\eta u_t.
$$

Think of $u_t$ as a running push: gradients pointing consistently in one direction accumulate, while opposing gradients partly cancel. This implementation uses the unnormalized sum above, not an average weighted by $1-\mu$.

### Implementation

```python
import numpy as np


def sgd_momentum(params, grads, velocity, lr=0.01, momentum=0.9):
    """
    params:   (D,) parameter vector
    grads:    (D,) current gradients
    velocity: (D,) previous momentum buffer

    Returns:
        updated_params, updated_velocity
    """

    velocity = momentum * velocity + grads
    params -= lr * velocity
    return params, velocity
```

### Example

```python
params = np.array([1.0, 2.0])
grads = np.array([0.5, -0.5])
velocity = np.zeros(2)
params, velocity = sgd_momentum(params, grads, velocity, lr=0.1, momentum=0.9)
print(params, velocity)
# [0.95 2.05] [ 0.5 -0.5]
```

The functions in the examples are defined in `optimizer.py`. SGD updates `params` in place, so callers holding the same array see the change.

## Adam: adaptive scaling and numerical considerations

Adam maintains exponentially weighted estimates of the gradient and its square:

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
$$

The first moment $m_t$ tracks direction; $v_t$ tracks recent gradient magnitude separately for each coordinate. Since both start at zero, their early values are pulled toward zero. Adam corrects for this initialization and updates parameters with

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t},\qquad
\theta_t=\theta_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

For a constant gradient, the correction exactly removes the zero-initialization factor; for changing gradients, it corrects that initialization effect but does not guarantee an unbiased estimate of the current gradient. The denominator makes coordinates with larger recent squared gradients take smaller steps relative to their first moment. Epsilon is added **outside** the square root to avoid division by zero and moderate very small denominators; it does not prevent overflow in $g_t^2$. The source has no separate numerically stable variant.

### Implementation

```python
def adam_optmization(
    params, grads, m, v, t,
    lr=0.001,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8
):
    """
    params: Parameter vector, shape (D,)
    grads:  Current gradients, shape (D,)
    m:      Previous first-moment estimate, shape (D,)
    v:      Previous second-moment estimate, shape (D,)
    t:      Current optimization step, starting at 1

    Returns:
        updated_params, updated_m, updated_v
    """

    if not isinstance(t, (int, np.integer)) or t < 1:
        raise ValueError("t must be an integer starting at 1")

    # 1. Update first moment
    m = beta1 * m + (1 - beta1) * grads

    # 2. Update second moment
    v = beta2 * v + (1 - beta2) * grads**2

    # 3. Bias correction
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    # 4. Update parameters
    params -= lr * m_hat / (np.sqrt(v_hat) + eps)

    return params, m, v
```

### Example

```python
params = np.array([1.0, 2.0])
grads = np.array([0.5, -0.5])
m = np.zeros(2)
v = np.zeros(2)
params, m, v = adam_optmization(params, grads, m, v, t=1, lr=0.1)
print(params, m, v)
# [0.9 2.1] [ 0.05 -0.05] [0.00025 0.00025]
```

Here $\hat m_1=g_1$ and $\hat v_1=g_1^2$, so each initial update is approximately $0.1$ in the gradient's direction. Adam updates `params` in place; the printed values are rounded.

### AdamW

AdamW applies weight decay directly to the old parameters, independently of Adam's moment calculations:

$$
\theta_t=\theta_{t-1}
-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
-\eta\lambda\theta_{t-1}.
$$

```python
def adamw_optmization(
    params, grads, m, v, t,
    lr=0.001,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    weight_decay=0.01
):
    """
    params: Previous parameters, shape (D,)
    grads: Current gradients, shape (D,)
    m: Previous first moment
    v: Previous second moment
    t: Current step, starting at 1

    Returns:
        updated_params, updated_m, updated_v
    """

    if not isinstance(t, (int, np.integer)) or t < 1:
        raise ValueError("t must be an integer starting at 1")

    # 1. Update first moment
    m = beta1 * m + (1 - beta1) * grads

    # 2. Update second moment
    v = beta2 * v + (1 - beta2) * grads ** 2

    # 3. Bias correction
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    # 4. Calculate Adam update
    adam_update = lr * m_hat / (np.sqrt(v_hat) + eps)

    # 5. Apply decoupled weight decay
    params = params - adam_update - lr * weight_decay * params

    return params, m, v
```

For example, with zero gradients and zero moment buffers, `lr=0.1` and `weight_decay=0.1`, parameters `[1.0, -2.0]` become `[0.99, -1.98]`. Unlike the other two functions, `adamw_optmization` returns a new parameter array rather than modifying the input in place.

## Follow-up questions

**Why is bias correction necessary?** The moments begin at zero, so early exponential averages have not yet accumulated their full weight. Dividing by $1-\beta_1^t$ and $1-\beta_2^t$ corrects this zero-initialization effect, especially in early steps.

**Why maintain squared gradients?** Their moving average measures recent gradient magnitude for each coordinate. Its square root scales the update, reducing the relative step where recent gradients have been large.

**What is the difference between Adam and AdamW?** Adding an L2 penalty to Adam's loss adds $\lambda\theta$ to the gradient, which enters both moment estimates and is adaptively scaled. AdamW instead applies decay directly to the parameters, separate from those moments. Plain Adam in the source applies no weight decay.

**What happens when the gradient is zero?** If the first moment has accumulated a nonzero value, it decays but can still move the parameters. With zero current gradient and zero first moment, Adam's gradient-based update is zero; AdamW can still change nonzero parameters through weight decay.
