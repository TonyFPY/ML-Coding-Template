
import numpy as np

def layer_norm(x, gamma, beta, eps=1e-5):
    # x: (B, D)
    # gamma: (D,)
    # beta: (D,)
    # TODO: Implement layer normalization
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    x_norm = (x - mean) /np.sqrt(var + eps)
    output = x_norm * gamma + beta
    return output

def batch_norm(
    x, gamma, beta, running_mean, running_var,
    training=True, momentum=0.1, eps=1e-5
):
    # x: (B, D)
    # gamma: (D,)
    # beta: (D,)

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

def rms_norm(x, gamma, eps=1e-5):
    # x: (B, D)
    # gamma: (D,)

    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    x_norm = x / rms
    output = x_norm * gamma
    return output


x = np.array([
    [1., 2., 3.],
    [4., 5., 6.]
])

gamma = np.ones(3)
beta = np.zeros(3)

out = layer_norm(x, gamma, beta)

print(out)
print("Mean:", np.mean(out, axis=-1))
print("Variance:", np.var(out, axis=-1))

x = np.array([
    [1., 2., 3.],
    [4., 5., 6.]
])

gamma = np.ones(3)
beta = np.zeros(3)

running_mean = np.zeros(3)
running_var = np.ones(3)

out, running_mean, running_var = batch_norm(
    x, gamma, beta,
    running_mean, running_var,
    training=True,
    momentum=0.1
)

print("Output:", out)
print("Mean:", np.mean(out, axis=0))
print("Variance:", np.var(out, axis=0))
print("Running mean:", running_mean)
print("Running variance:", running_var)

x_test = np.array([[1., 2., 3.]])

out, running_mean, running_var = batch_norm(
    x_test, gamma, beta,
    running_mean, running_var,
    training=False
)

print("Inference output:", out)
print("Running mean:", running_mean)
print("Running variance:", running_var)

x = np.array([
    [1., 2., 3.],
    [-3., 0., 4.]
])

gamma = np.ones(3)

out = rms_norm(x, gamma)

print(out)
print("RMS:", np.sqrt(np.mean(out ** 2, axis=-1)))

x = np.array([
    [3., 3., 3.],
    [3., 3., 3.]
])

gamma = np.ones(3)
beta = np.zeros(3)

print("LayerNorm:", layer_norm(x, gamma, beta))

out, _, _ = batch_norm(
    x, gamma, beta,
    np.zeros(3), np.ones(3)
)
print("BatchNorm:", out)

print("RMSNorm:", rms_norm(x, gamma))