import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    """
    """
    X * w = y
    => X^T * X * w = X^T * y
    => w = (X^T * X)^(-1) * X^T * y
    """
    X = np.asarray(X)
    y = np.asarray(y)
    XT = np.linalg.matrix_transpose(X)
    return np.linalg.inv(XT @ X) @ XT @ y
    