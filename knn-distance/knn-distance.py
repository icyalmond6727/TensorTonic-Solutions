import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return k nearest neighbor indices.
    """
    X_train = np.asarray(X_train)
    if X_train.ndim == 1: X_train = X_train.reshape(-1, 1)
    X_test = np.asarray(X_test)
    if X_test.ndim == 1: X_test = X_test.reshape(-1, 1)
    distances = np.empty(shape = (len(X_train), ))
    if len(X_test) == 0: return np.empty(shape = (0, k), dtype = int)
    ans = []
    for i in range(0, len(X_test)):
        if len(X_train) == 0:
            ans.append([-1] * k)
            continue
        distances = np.linalg.norm(X_train - X_test[i], axis = 1)
        distances = np.argsort(distances).tolist()
        if len(distances) < k: distances.extend([-1] * (k - len(distances)))
        ans.append(distances[ : k])
    return np.asarray(ans)