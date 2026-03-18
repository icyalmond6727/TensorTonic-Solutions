import numpy as np

def giny(y):
    if len(y) == 0: return float(0)
    labels, counts = np.unique(y, return_counts = True)
    counts = counts[counts > 0]
    p = counts / len(y)
    p = p ** 2
    return 1 - p.sum()

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    if len(y_left) == 0 and len(y_right) == 0: return float(0)
    y_left = np.asarray(y_left)
    y_right = np.asarray(y_right)
    p_left = len(y_left) / (len(y_left) + len(y_right))
    p_right = 1 - p_left
    return giny(y_left) * p_left + giny(y_right) * p_right