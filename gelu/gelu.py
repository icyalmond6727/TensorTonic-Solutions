import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: list or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    a = np.asarray(x, dtype = float)
    vectorized_erf = np.vectorize(math.erf)
    a = 0.5 * a * (1 + vectorized_erf(a / math.sqrt(2)))
    return a
