import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    labels, counts = np.unique(y, return_counts = True)
    ans = float(0)
    for i in range(0, len(counts)):
        if counts[i] == 0: continue
        p = counts[i] / len(y)
        ans -= p * np.log2(p)
    return ans