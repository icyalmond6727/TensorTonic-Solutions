import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    if len(y) == 0: return float(0)
    labels, counts = np.unique(y, return_counts = True)
    counts = counts[counts > 0]
    p = counts / len(y)
    p = -(p * np.log2(p))
    return p.sum()