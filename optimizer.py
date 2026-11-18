'''
Objectives
- Implement SGD with momentum.
- Implement Adam with bias correction.
- Explain the difference between Adam and AdamW.
'''

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

'''
Follow-up questions
After implementing Adam, the interviewer may ask:
1. Why is bias correction necessary? Explain how initializing both moment estimates at zero introduces a bias toward zero in early iterations.
2. Why does Adam maintain squared gradients? Explain how the second moment controls the per-parameter update magnitude.
3. What is the difference between Adam and AdamW? Explain the distinction between incorporating an L2 penalty into the gradient and applying decoupled weight decay.
4. What happens when the gradient is zero? Discuss how accumulated first-moment estimates can still produce an update even if the current gradient is zero.
'''

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


if __name__ == "__main__":
    params = np.array([1.0, 2.0])
    grads = np.array([0.5, -0.5])
    velocity = np.zeros(2)

    params, velocity = sgd_momentum(
        params, grads, velocity, lr=0.1, momentum=0.9
    )

    print(params)    # [0.95, 2.05]
    print(velocity)  # [0.5, -0.5]

    params = np.array([1.0, 2.0])
    grads = np.array([0.5, -0.5])

    m = np.zeros(2)
    v = np.zeros(2)

    params, m, v = adam_optmization(
        params, grads, m, v,
        t=1, lr=0.1
    )

    print("Parameters:", params)
    print("First moment:", m)
    print("Second moment:", v)

    params = np.array([1.0, 2.0])
    grads = np.array([0.5, -0.5])

    m = np.zeros(2)
    v = np.zeros(2)

    params, m, v = adamw_optmization(
        params, grads, m, v,
        t=1,
        lr=0.1,
        weight_decay=0.01
    )

    print(params)
    print(m)
    print(v)

    params = np.array([1.0, -2.0])
    grads = np.zeros(2)
    m = np.zeros(2)
    v = np.zeros(2)

    params, m, v = adamw_optmization(
        params, grads, m, v,
        t=1,
        lr=0.1,
        weight_decay=0.1
    )

    print(params)
