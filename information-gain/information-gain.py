import numpy as np

def _entropy(y):
    """
    Helper: Compute Shannon entropy (base 2) for labels y.
    """
    if len(y) == 0: return float(0)
    labels, counts = np.unique(y, return_counts = True)
    counts = counts[counts > 0]
    p = counts / len(y)
    p = -(p * np.log2(p))
    return p.sum()

def information_gain(y, split_mask):
    """
    Compute Information Gain of a binary split on labels y.
    Use the _entropy() helper above.
    """
    if len(y) == 0: return float(0)
    y = np.asarray(y)
    left_y = y[split_mask]
    right_y = y[~split_mask]
    return _entropy(y) - (len(left_y) / len(y) * _entropy(left_y) + len(right_y) / len(y) * _entropy(right_y))
