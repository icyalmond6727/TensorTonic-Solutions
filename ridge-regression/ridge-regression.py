def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    XT = np.linalg.matrix_transpose(X)
    return np.linalg.inv(XT @ X + lam * np.identity(X.shape[1])) @ XT @ y