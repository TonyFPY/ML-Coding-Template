import numpy as np


def dropout(x, p=0.5, training=True):
    if not 0 <= p < 1:
        raise ValueError("p must satisfy 0 <= p < 1")
    if not training or p == 0:
        return x

    mask = np.random.random(x.shape) >= p
    return x * mask / (1 - p)


if __name__ == "__main__":
    # Test 1: Training mode
    np.random.seed(42)

    x = np.ones((4, 5))
    out = dropout(x, p=0.5, training=True)

    print(out)
    # Values should be either 0 or 2.


    # Test 2: Evaluation mode
    out = dropout(x, p=0.5, training=False)

    print(out)
    # All values should remain 1.


    # Test 3: No dropout
    out = dropout(x, p=0.0, training=True)

    print(out)
    # All values should remain 1.


    # Test 4: Expected activation
    x = np.ones(100000)
    out = dropout(x, p=0.3, training=True)

    print(np.mean(out))
    # Approximately 1.0 (sampling variation expected)


'''
Follow-up questions
Once your implementation works, you should be able to explain:
- Why do we scale by \(1/(1-p)\)?
- Why use inverted dropout instead of standard dropout? Inverted dropout scales during training, making inference a direct identity operation.
-  Why apply dropout only during training? Randomly removing activations during inference would introduce unnecessary stochasticity and distort predictions.
- What happens when p approaches 1? Very few activations survive, and those retained are multiplied by a large scaling factor, increasing output variance.
- Where would you apply dropout in an MLP? Typically to hidden activations during training, often after a nonlinear activation.
'''
