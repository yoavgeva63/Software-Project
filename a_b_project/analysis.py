import sys
import numpy as np
from math import sqrt
import symnmf as symnmf_c
from kmeans import kmeans_algorithm

# Constants declaration
MAX_ITER = 300
EPS = 1e-4

def error():
    """Print unified error and exit."""
    print("An Error Has Occurred")
    sys.exit(1)

def load_data(path):
    """Load data from a file into a numpy array."""
    try:
        return np.loadtxt(path, delimiter=',')
    except Exception:
        error()

# Helper functions
def avg_value_matrix(M):
    """Average of all entries in a matrix M."""
    return np.mean(M)

def init_H(k, n, avgW, seed=1234):
    """
    Random nonnegative H0 (n×k) with entries ~ U[0, 2*sqrt(avgW/k)].
    Uses a fixed seed as demanded.
    """
    np.random.seed(seed)
    scale = 2 * np.sqrt(avgW / k)
    return np.random.uniform(0, scale, size=(n, k)).tolist()

# Implementation of silhouette_score
def pairwise_distances(data):
    """Return full n×n Euclidean distance matrix."""
    n = data.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(data[i] - data[j])
            D[i, j] = D[j, i] = dist
    return D

def silhouette_score(data, labels):
    """
    Manual implementation of mean silhouette score (Euclidean).
    s(i)=(b-a)/max(a,b). Returns float in [-1,1].
    """
    n = len(data)
    D = pairwise_distances(data)
    unique_labels = np.unique(labels)
    
    if len(unique_labels) <= 1:
        return 0.0

    # Create a dictionary to hold indices for each cluster
    clusters = {label: np.where(labels == label)[0] for label in unique_labels}
    
    silhouette_vals = []
    for i in range(n):
        my_cluster_label = labels[i]
        my_cluster_indices = clusters[my_cluster_label]

        # Calculate a(i) - mean intra-cluster distance
        if len(my_cluster_indices) <= 1:
            a = 0.0
        else:
            a = np.mean([D[i, j] for j in my_cluster_indices if i != j])

        # Calculate b(i) - mean nearest-cluster distance
        min_other_dist = float('inf')
        for label, indices in clusters.items():
            if label == my_cluster_label:
                continue
            other_dist = np.mean([D[i, j] for j in indices])
            if other_dist < min_other_dist:
                min_other_dist = other_dist
        b = min_other_dist
        
        # Calculate silhouette coefficient for point i
        if max(a, b) == 0:
            s_i = 0.0
        else:
            s_i = (b - a) / max(a, b)
        silhouette_vals.append(s_i)
        
    return np.mean(silhouette_vals)

def symnmf_labels(data, k):
    """Get cluster labels from the SymNMF algorithm."""
    try:
        data_list = data.tolist()
        A = symnmf_c.sym(data_list)
        D = symnmf_c.ddg(A)
        W = symnmf_c.norm(A, D)

        avgW = avg_value_matrix(W)
        H0 = init_H(k, len(data), avgW)
        
        H = symnmf_c.symnmf(W, H0, MAX_ITER, EPS)
        
        return np.argmax(H, axis=1)
    except Exception:
        error()

def kmeans_labels(data, k):
    """Get cluster labels from the K-Means algorithm."""
    try:
        data_list = data.tolist()
        centroids = kmeans_algorithm(k, data_list, MAX_ITER, EPS)
        
        centroids_np = np.array(centroids)
        labels = np.zeros(len(data), dtype=int)
        for i, point in enumerate(data):
            distances = np.linalg.norm(point - centroids_np, axis=1)
            labels[i] = np.argmin(distances)
        return labels
    except Exception:
        error()

def main():
    """
    CLI: k, input_file -> compare silhouette of SymNMF vs KMeans.
    """
    if len(sys.argv) != 3:
        error()

    try:
        k = int(sys.argv[1])
        path = sys.argv[2]
    except ValueError:
        error()

    data = load_data(path)
    if not (1 < k < len(data)):
        error()
        
    # Get labels from both algorithms
    nmf_labs = symnmf_labels(data, k)
    km_labs = kmeans_labels(data, k)
    
    # Calculate silhouette scores
    s_nmf = silhouette_score(data, nmf_labs)
    s_km = silhouette_score(data, np.array(km_labs))
    
    # Print results
    print(f"nmf: {s_nmf:.4f}")
    print(f"kmeans: {s_km:.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        error()
