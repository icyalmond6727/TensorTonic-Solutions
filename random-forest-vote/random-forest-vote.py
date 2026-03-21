import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    predictions = np.asarray(predictions)
    results = []
    for j in range(0, len(predictions[0])):
        labels, counts = np.unique(predictions[ : , j], return_counts = True)
        majority_label = labels[np.argmax(counts)]
        results.append(majority_label)
    return results