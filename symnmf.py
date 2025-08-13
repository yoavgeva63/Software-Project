from math import sqrt
import sys
import csv
import random
import symnmf  # C extension: sym, ddg, norm, symnmf

def error():
    """Print unified error and exit with non-zero status."""
    print("An Error Has Occurred")
    sys.exit(1)

def load_csv(path):
    """Load CSV into a list[list[float]]. No headers, all rows same length."""
    try:
        with open(path, "r", newline="") as f:
            rows = []
            for row in csv.reader(f):
                if not row:
                    continue
                rows.append([float(x) for x in row])
            if not rows:
                error()
            # shape consistency
            m = len(rows[0])
            if any(len(r) != m for r in rows):
                error()
            return rows
    except Exception:
        error()

def print_matrix(M):
    """Print matrix with 4 decimals, comma-separated (no trailing comma)."""
    try:
        for row in M:
            print(",".join(f"{float(x):.4f}" for x in row))
    except Exception:
        error()

# ------------------------ math helpers ------------------------

def avg_value_matrix(M):
    """Average of all entries in M (0 if empty)."""
    if not M:
        return 0.0
    total = 0.0
    count = 0
    for r in M:
        for v in r:
            total += float(v)
            count += 1
    return (total / count) if count else 0.0

def init_H(k, n, avgW, seed=1234):
    """
    Random nonnegative H0 (n×k) with entries ~ U[0, 2*sqrt(avgW/k)].
    Uses a fixed seed for reproducibility.
    """
    if k <= 0 or n <= 0:
        error()
    scale = 2.0 * sqrt(avgW / k) if k > 0 else 0.0
    rng = random.Random(seed)
    return [[rng.uniform(0.0, scale) for _ in range(k)] for _ in range(n)]

# ------------------------ args & main ------------------------

def parse_args(argv):
    """
    Expect: python symnmf.py k goal input_file
    goal in {"sym","ddg","norm","symnmf"}.
    For symnmf, k must satisfy 1 ≤ k < n.
    """
    if len(argv) != 4:
        error()
    try:
        k = int(argv[1])
        goal = argv[2]
        path = argv[3]
        data = load_csv(path)
    except Exception:
        error()
    n = len(data)
    if goal not in {"sym", "ddg", "norm", "symnmf"}:
        error()
    if goal == "symnmf" and (k <= 0 or k >= n):
        error()
    return k, goal, data

def main():
    """CLI: compute sym/ddg/norm, or run SymNMF with simple random init."""
    eps = 1e-4
    max_iter = 300
    try:
        k, goal, data = parse_args(sys.argv)

        if goal == "sym":
            A = symnmf.sym(data)
            print_matrix(A)
            return

        if goal == "ddg":
            D = symnmf.ddg(data)
            print_matrix(D)
            return

        if goal == "norm":
            W = symnmf.norm(data)
            print_matrix(W)
            return

        # goal == "symnmf"
        W = symnmf.norm(data)
        avgW = avg_value_matrix(W)
        H0 = init_H(k, len(data), avgW) 
        H = symnmf.symnmf(W, H0, max_iter, eps)
        print_matrix(H)

    except SystemExit:
        raise
    except Exception:
        error()

if __name__ == "__main__":
    main()
