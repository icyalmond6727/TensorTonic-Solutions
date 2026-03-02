import numpy as np

def calc(x, alpha):
    if x >= 0:
        return x
    else:
        return alpha * x

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    a = np.asarray(x, dtype = float)
    return np.vectorize(calc)(a, alpha)