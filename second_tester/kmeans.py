import sys
from math import inf, sqrt

def error():
    """Prints a unified error message and exits."""
    print("An Error Has Occurred")
    sys.exit(1)

def euclidean_distance(a, b):
    """Calculates the Euclidean distance between two vectors."""
    s = 0.0
    for i in range(len(a)):
        d = a[i] - b[i]
        s += d * d
    return sqrt(s)

def compute_mean(vectors):
    """Computes the coordinate-wise mean of a list of vectors."""
    if not vectors: return []
    dim = len(vectors[0])
    mean = [0.0] * dim
    for v in vectors:
        for i in range(dim):
            mean[i] += v[i]
    for i in range(dim):
        mean[i] /= len(vectors)
    return mean

def find_closest_centroid(vector, centroids):
    """Returns the index of the closest centroid to a given vector."""
    best_i, min_d = 0, inf
    for i, c in enumerate(centroids):
        d = euclidean_distance(vector, c)
        if d < min_d:
            best_i, min_d = i, d
    return best_i

def assign_points(data, centroids, k):
    """Assigns each data point to the nearest centroid."""
    buckets = [[] for _ in range(k)]
    for v in data:
        idx = find_closest_centroid(v, centroids)
        buckets[idx].append(v)
    return buckets

def kmeans_algorithm(k, data, max_iters, eps):
    """
    Performs the K-Means clustering algorithm.
    
    Args:
        k (int): Number of clusters.
        data (list): List of data points (vectors).
        max_iters (int): Maximum number of iterations.
        eps (float): Convergence threshold.
    
    Returns:
        list: A list of the final k centroids.
    """
    try:
        # Basic validation
        if not (1 <= k <= len(data)): error()
        data_float = [[float(x) for x in row] for row in data]
        centroids = data_float[:k] # Simple deterministic initialization

        for _ in range(max_iters):
            buckets = assign_points(data_float, centroids, k)
            max_delta = 0.0
            for i in range(k):
                if not buckets[i]: continue
                new_c = compute_mean(buckets[i])
                delta = euclidean_distance(new_c, centroids[i])
                if delta > max_delta: max_delta = delta
                centroids[i] = new_c
            if max_delta < eps:
                break
        return centroids
    except Exception:
        error()
