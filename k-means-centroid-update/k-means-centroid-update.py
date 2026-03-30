import numpy as np

def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    points = np.asarray(points)
    assignments = np.asarray(assignments)
    sums = np.zeros((k, points.shape[1]))
    counts = np.zeros(k)
    for i in range(len(points)):
        sums[assignments[i]] += points[i]
        counts[assignments[i]] += 1
    counts = np.maximum(counts, 1)
    ans = []
    for i in range(k):
        ans.append((sums[i] / counts[i]).tolist())
    return ans