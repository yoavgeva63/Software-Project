import sys
import numpy as np
import symnmf as symnmf_c
from kmeans import kmeans_algorithm
from sklearn.metrics import silhouette_score

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
        # Use atleast_2d to handle single-column files correctly
        return np.atleast_2d(np.loadtxt(path, delimiter=','))
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
    
    # Calculate silhouette scores using sklearn
    s_nmf = silhouette_score(data, nmf_labs)
    s_km = silhouette_score(data, km_labs)
    
    # Print results
    print(f"nmf: {s_nmf:.4f}")
    print(f"kmeans: {s_km:.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        error()