import sys
import numpy as np
import symnmf as symnmf_c

def error():
    """Print unified error and exit."""
    print("An Error Has Occurred")
    sys.exit(1)

def load_data(path):
    """Load data from a file into a list of lists of floats."""
    try:
        return np.loadtxt(path, delimiter=',').tolist()
    except Exception:
        error()

def print_matrix(M):
    """Print matrix with 4 decimals, comma-separated."""
    try:
        for row in M:
            print(",".join(f"{x:.4f}" for x in row))
    except Exception:
        error()

def init_H(k, n, avgW, seed=1234):
    """
    Random nonnegative H0 (n×k) with entries ~ U[0, 2*sqrt(avgW/k)].
    Uses a fixed seed for reproducibility as required.
    """
    np.random.seed(seed)
    scale = 2 * np.sqrt(avgW / k)
    return np.random.uniform(0, scale, size=(n, k)).tolist()

def avg_value_matrix(M):
    """Average of all entries in M."""
    return np.mean(M)

def main():
    """CLI: compute sym/ddg/norm, or run SymNMF."""
    if len(sys.argv) != 4:
        error()

    try:
        k = int(sys.argv[1])
        goal = sys.argv[2]
        file_path = sys.argv[3]
    except (ValueError, IndexError):
        error()

    data = load_data(file_path)
    n = len(data)

    if goal not in {"symnmf", "sym", "ddg", "norm"}:
        error()

    try:
        if goal == "sym":
            A = symnmf_c.sym(data)
            print_matrix(A)
        elif goal == "ddg":
            A = symnmf_c.sym(data)
            D = symnmf_c.ddg(A)
            print_matrix(D)
        elif goal == "norm":
            A = symnmf_c.sym(data)
            D = symnmf_c.ddg(A)
            W = symnmf_c.norm(A, D)
            print_matrix(W)
        elif goal == "symnmf":
            if not (1 < k < n):
                error()
            
            A = symnmf_c.sym(data)
            D = symnmf_c.ddg(A)
            W = symnmf_c.norm(A, D)
            
            avgW = avg_value_matrix(W)
            H0 = init_H(k, n, avgW)
            
            H = symnmf_c.symnmf(W, H0, 300, 1e-4)
            print_matrix(H)

    except Exception:
        error()

if __name__ == "__main__":
    main()