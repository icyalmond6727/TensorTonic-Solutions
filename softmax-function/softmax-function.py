import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    a = np.asarray(x)
    ex = np.exp(a - np.max(a, axis = -1, keepdims = True))
    return ex / np.sum(ex, axis = -1, keepdims = True)