import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    a = np.asarray(x)
    return a / (1 + np.exp(-a))