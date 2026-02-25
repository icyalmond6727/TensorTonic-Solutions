import numpy as np

def tanh(x):
    a = np.asarray(x)
    a = (np.exp(a) - np.exp(-a)) / (np.exp(a) + np.exp(-a))
    return a