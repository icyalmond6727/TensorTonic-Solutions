import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    labels, counts = np.unique(y_train, return_counts = True)
    majority_label = labels[np.argmax(counts)]
    if len(X_test) == 0:
       return np.asarray([], dtype = int)
    else:
        return np.full(shape = (len(X_test),), fill_value = majority_label, dtype = int)