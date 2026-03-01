import numpy as np

def calc(x, lam, alpha):
    if x > 0:
        return lam * x
    else: 
        return lam * alpha * (np.exp(x) - 1)

def selu(x, lam=1.0507009873554804934193349852946, alpha=1.6732632423543772848170429916717):
    """
    Apply SELU activation element-wise.
    Returns a list of floats rounded to 4 decimal places.
    """
    a = np.asarray(x, dtype = float)
    a = np.vectorize(calc)(a, lam, alpha)
    return a
