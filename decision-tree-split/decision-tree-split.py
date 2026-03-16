import numpy as np

def gini(y):
    labels, counts = np.unique(y, return_counts = True)
    counts = counts[counts > 0]
    p = counts / len(y)
    p = p**2
    return 1 - p.sum()

def information_gain(y, split_mask):
    left_y = y[split_mask]
    right_y = y[~split_mask]
    if len(left_y) == 0 or len(right_y) == 0: return 0
    return gini(y) - (len(left_y) / len(y) * gini(left_y) + len(right_y) / len(y) * gini(right_y))

def decision_tree_split(X, y):
    """
    Find the best feature and threshold to split the data.
    """
    y = np.asarray(y)
    
    best_gain = 0
    best_feature = 0
    best_threshold = 0
    
    for feature in range(0, len(X[0])):
        temp = sorted(set([X[sample][feature] for sample in range(len(X))]))
        
        for i in range(0, len(temp) - 1):
            threshold = (temp[i] + temp[i + 1]) / 2
            split_mask = np.array([X[sample][feature] <= threshold for sample in range(len(X))])
            gain = information_gain(y, split_mask)
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
                
    return [best_feature, best_threshold]