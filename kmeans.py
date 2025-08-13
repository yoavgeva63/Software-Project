import sys
from math import inf, sqrt

def error():
    """Print unified error and exit."""
    print("An Error Has Occurred")
    sys.exit(1)

# ---------- distance & stats ----------

def euclidean_distance(a, b):
    """Return Euclidean distance between two equal-length sequences."""
    if len(a) != len(b):
        error()
    s = 0.0
    for i in range(len(a)):
        d = float(a[i]) - float(b[i])
        s += d * d
    return sqrt(s)

def compute_mean(vectors):
    """Return coordinate-wise mean of non-empty list of equal-length vectors."""
    if not vectors:
        error()
    dim = len(vectors[0])
    mean = [0.0] * dim
    for v in vectors:
        if len(v) != dim:
            error()
        for i in range(dim):
            mean[i] += float(v[i])
    for i in range(dim):
        mean[i] /= len(vectors)
    return mean

# ---------- core helpers ----------

def validate_data(k, data, max_iters, eps):
    """Validate K, data shape/types, and parameters."""
    if not isinstance(data, list) or not data:
        error()
    dim = len(data[0])
    if dim == 0:
        error()
    for row in data:
        if not isinstance(row, list) or len(row) != dim:
            error()
        # Coerce check to float without mutating input
        for x in row:
            float(x)
    if not (1 <= k <= len(data)):
        error()
    if not (1 <= max_iters <= 1000):
        error()
    if eps <= 0:
        error()
    return dim

def find_closest_centroid(vector, centroids):
    """Return index of nearest centroid to 'vector'."""
    best_i, best_d = 0, inf
    for i, c in enumerate(centroids):
        d = euclidean_distance(vector, c)
        if d < best_d:
            best_i, best_d = i, d
    return best_i

def assign_points(data, centroids, k):
    """Assign each data point to nearest centroid; returns list of k buckets."""
    buckets = [[] for _ in range(k)]
    for v in data:
        idx = find_closest_centroid(v, centroids)
        buckets[idx].append(v)
    return buckets

# ---------- public API ----------

def kmeans_algorithm(k, data, max_iters, eps):
    """
    - k: number of clusters (1..n)
    - data: list[list[float]] with consistent dimensionality
    - max_iters: 1..1000
    - eps: convergence threshold (>0), stop when centroid shift <= eps
    Returns a list of k centroids (each a list[float]).
    """
    try:
        validate_data(k, data, max_iters, eps)
        centroids = [list(data[i]) for i in range(k)]  # simple deterministic init

        for _ in range(max_iters):
            buckets = assign_points(data, centroids, k)
            converged = True
            for i in range(k):
                if not buckets[i]:
                    # keep centroid as-is if its bucket is empty
                    continue
                new_c = compute_mean(buckets[i])
                if euclidean_distance(new_c, centroids[i]) > eps:
                    converged = False
                centroids[i] = new_c
            if converged:
                break
        return centroids
    except Exception:
        error()
