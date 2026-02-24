import numpy as np
import math

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    a = np.asarray(x, dtype = float)
    a = 1 / (1 + np.exp(-a))
    return a
    pass