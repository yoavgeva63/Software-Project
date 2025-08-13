import sys
from math import sqrt
# fixed hyperparameters
MAX_ITER = 300
EPS = 1e-4

# guarded imports
try:
    import symnmf
    from kmeans import kmeans_algorithm, euclidean_distance
except Exception:
    print("An Error Has Occurred")
    sys.exit(1)

def error():
    """Print unified error and exit."""
    print("An Error Has Occurred")
    sys.exit(1)

def read_dataset(path):
    """
    Load a rectangular 2D dataset from path (CSV or whitespace).
    Skips blank lines. Returns list[list[float]] or exits on error.
    """
    rows = []
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",") if "," in line else line.split()
                rows.append([float(x) for x in parts])
    except Exception:
        error()
    if not rows or any(len(r) != len(rows[0]) for r in rows):
        error()
    return rows

def kmeans_labels(data, k):
    """
    Labels via K-Means: run kmeans_algorithm, then assign each point to
    its nearest centroid by Euclidean distance.
    """
    try:
        cents = kmeans_algorithm(k, data, MAX_ITER, EPS)
    except Exception:
        error()
    if not cents or len(cents) != k:
        error()
    labels = []
    for v in data:
        best, bestd = 0, float("inf")
        for j, c in enumerate(cents):
            d = euclidean_distance(v, c)
            if d < bestd:
                best, bestd = j, d
        labels.append(best)
    return labels

def build_W(data):
    """
    Compute normalized similarity W using the C extension:
    A = sym(data), D = ddg(A), W = norm(A, D).
    """
    A = symnmf.sym(data)
    if A is None:
        error()
    D = symnmf.ddg(A)
    if D is None:
        error()
    W = symnmf.norm(A, D)
    if W is None:
        error()
    return W

def symnmf_labels(data, k):
    """
    Run SymNMF with random H0 scaled by W statistics.
    Labels are argmax over columns for each row of H.
    """
    W = build_W(data)
    H0 = symnmf.init_H(k, len(data), symnmf.avg_value_matrix(W))
    H = symnmf.symnmf(W, H0, MAX_ITER, EPS)
    if H is None:
        error()
    labels = []
    for row in H:
        jmax, vmax = 0, row[0]
        for j in range(1, k):
            if row[j] > vmax:
                jmax, vmax = j, row[j]
        labels.append(jmax)
    return labels

def pairwise_distances(data):
    """Return full n×n Euclidean distance matrix."""
    n, d = len(data), len(data[0])
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = 0.0
            for t in range(d):
                diff = float(data[i][t]) - float(data[j][t])
                s += diff * diff
            val = sqrt(s)
            D[i][j] = D[j][i] = val
    return D

def silhouette_score(data, labels):
    """
    Mean silhouette (Euclidean).
    a(i): mean intra-cluster distance (0 for singletons).
    b(i): min mean distance to other clusters.
    s(i)=(b-a)/max(a,b). Returns float in [-1,1].
    """
    n = len(data)
    D = pairwise_distances(data)
    clusters = {}
    for i, c in enumerate(labels):
        clusters.setdefault(c, []).append(i)
    if len(clusters) <= 1:
        return 0.0
    total, count = 0.0, 0
    for i, ci in enumerate(labels):
        same = clusters[ci]
        if len(same) <= 1:
            si = 0.0
        else:
            a = sum(D[i][j] for j in same if j != i) / (len(same) - 1)
            b = float("inf")
            for c, idxs in clusters.items():
                if c == ci:
                    continue
                b = min(b, sum(D[i][j] for j in idxs) / len(idxs))
            si = 0.0 if b == 0.0 else (b - a) / max(a, b)
        total += si; count += 1
    return (total / count) if count else 0.0

def main():
    """CLI: k, input_file -> compare silhouette of SymNMF vs KMeans."""
    if len(sys.argv) != 3:
        error()
    try:
        k = int(sys.argv[1]); path = sys.argv[2]
    except Exception:
        error()
    data = read_dataset(path)
    if not (1 <= k <= len(data)):
        error()
    try:
        labs_km = kmeans_labels(data, k)
        labs_nmf = symnmf_labels(data, k)
        s_nmf = silhouette_score(data, labs_nmf)
        s_km = silhouette_score(data, labs_km)
        print(f"nmf: {s_nmf:.4f}")
        print(f"kmeans: {s_km:.4f}")
    except SystemExit:
        raise
    except Exception:
        error()

if __name__ == "__main__":
    main()
