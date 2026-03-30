import numpy as np

def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """
    points = np.asarray(points)
    centroids = np.asarray(centroids)
    ans = []
    for i in range(len(points)):
        id = 0
        mn_d = float("inf")
        for j in range(len(centroids)):
            cur_d = np.linalg.norm(points[i] - centroids[j])
            if cur_d < mn_d:
                mn_d = cur_d
                id = j
        ans.append(id)
    return ans