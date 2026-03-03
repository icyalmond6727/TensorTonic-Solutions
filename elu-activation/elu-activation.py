import math

def calc(x, alpha):
    if x > 0:
        return x
    else: 
        return alpha * (math.exp(x) - 1)

def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    x = [calc(i, alpha) for i in x]
    return x