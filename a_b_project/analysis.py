import sys
import numpy as np
import symnmf as symnmf_c
from kmeans import kmeans_algorithm
from sklearn.metrics import silhouette_score

# Constants from the project description
MAX_ITER = 300
EPS = 1e-4

def error():
    """Prints a unified error message and exits."""
    print("An Error Has Occurred")
    sys.exit(1)

def load_data(path):
    """Loads data from a comma-delimited file into a numpy array."""
    try:
        return np.loadtxt(path, delimiter=',')
    except Exception:
        error()

def init_H(k, n, avgW, seed=1234):
    """
    Initializes a random non-negative H0 matrix (n x k).
    Entries are uniformly distributed in [0, 2*sqrt(avgW/k)].
    """
    np.random.seed(seed)
    scale = 2 * np.sqrt(avgW / k)
    return np.random.uniform(0, scale, size=(n, k)).tolist()

def symnmf_labels(data, k):
    """
    Computes cluster labels for the data using the SymNMF algorithm.
    Returns: A numpy array of cluster assignments.
    """
    data_list = data.tolist()
    A = symnmf_c.sym(data_list)
    D = symnmf_c.ddg(A)
    W = symnmf_c.norm(A, D)
    avgW = np.mean(W)
    H0 = init_H(k, len(data), avgW)
    H = symnmf_c.symnmf(W, H0, MAX_ITER, EPS)
    return np.argmax(H, axis=1)

def kmeans_labels(data, k):
    """
    Computes cluster labels for the data using the K-Means algorithm.
    Returns: A numpy array of cluster assignments.
    """
    data_list = data.tolist()
    centroids = kmeans_algorithm(k, data_list, MAX_ITER, EPS)
    centroids_np = np.array(centroids)
    # Assign each point to the closest centroid
    distances = np.linalg.norm(data[:, np.newaxis] - centroids_np, axis=2)
    return np.argmin(distances, axis=1)

def main():
    """
    Main entry point for comparing SymNMF and K-Means.
    CLI: python3 analysis.py <k> <input_file>
    """
    if len(sys.argv) != 3: error()
    try:
        k = int(sys.argv[1])
        path = sys.argv[2]
    except ValueError: error()

    data = load_data(path)
    if not (1 < k < len(data)): error()
        
    # Get labels and calculate silhouette scores using sklearn
    nmf_labs = symnmf_labels(data, k)
    s_nmf = silhouette_score(data, nmf_labs)
    
    km_labs = kmeans_labels(data, k)
    s_km = silhouette_score(data, km_labs)
    
    # Print results as required
    print(f"nmf: {s_nmf:.4f}")
    print(f"kmeans: {s_km:.4f}")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        error()
