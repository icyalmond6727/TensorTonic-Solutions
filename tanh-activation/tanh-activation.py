import numpy as np

def tanh(x):
    a = np.asarray(x)
    return (np.exp(a) - np.exp(-a)) / (np.exp(a) + np.exp(-a))