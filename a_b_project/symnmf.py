import sys
import numpy as np
import symnmf as symnmf_c

def error():
    """Prints a unified error message and exits the program."""
    print("An Error Has Occurred")
    sys.exit(1)

def load_data(path):
    """
    Loads data from a text file.
    Handles both comma-delimited and whitespace-delimited files.
    """
    try:
        return np.loadtxt(path, delimiter=',').tolist()
    except Exception:
        # Fallback for files without commas (e.g., single column)
        return np.loadtxt(path).tolist()

def print_matrix(M):
    """Prints a matrix with 4 decimal places, comma-separated."""
    for row in M:
        print(",".join(f"{x:.4f}" for x in row))

def init_H(k, n, avgW, seed=1234):
    """
    Initializes a random non-negative H0 matrix (n x k).
    Entries are uniformly distributed in [0, 2*sqrt(avgW/k)].
    Uses a fixed seed for reproducible results as required.
    """
    np.random.seed(seed)
    scale = 2 * np.sqrt(avgW / k)
    return np.random.uniform(0, scale, size=(n, k)).tolist()

def avg_value_matrix(M):
    """Calculates the average of all entries in matrix M."""
    return np.mean(M)

def main():
    """
    Command-line interface for the SymNMF tool.
    Performs a goal (sym, ddg, norm, symnmf) on an input data file.
    """
    if len(sys.argv) != 4: error()
    try:
        k = int(sys.argv[1])
        goal = sys.argv[2]
        file_path = sys.argv[3]
    except (ValueError, IndexError): error()
    if goal not in {"symnmf", "sym", "ddg", "norm"}: error()
    
    data = load_data(file_path)
    n = len(data)

    try:
        A = symnmf_c.sym(data)
        if goal == "sym": print_matrix(A); return
        
        D = symnmf_c.ddg(A)
        if goal == "ddg": print_matrix(D); return

        W = symnmf_c.norm(A, D)
        if goal == "norm": print_matrix(W); return

        if not (0 < k < n): error()
        avgW = avg_value_matrix(W)
        H0 = init_H(k, n, avgW)                    
        H = symnmf_c.symnmf(W, H0, 300, 1e-4)
        print_matrix(H)
    except Exception:
        error()

if __name__ == "__main__":
    main()
