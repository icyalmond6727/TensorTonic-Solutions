import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    labels, counts = np.unique(y_train, return_counts = True)
    majority_label = labels[np.argmax(counts)]
    if X_test.shape[0] == 0:
       return np.asarray([], dtype = int)
    predictions = np.full(shape = (X_test.shape[0], ), fill_value = majority_label, dtype = int)
    return predictions