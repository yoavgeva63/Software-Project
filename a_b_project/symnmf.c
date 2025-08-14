#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "symnmf.h"

/**
 * @brief A struct to hold all temporary matrices for the symNMF algorithm.
 */
typedef struct {
    double **H, **Hprev, **HT, **HHT, **WH, **HHTH;
} SymnmfMatrices;

/* Forward declaration */
static SymnmfMatrices* alloc_symnmf_matrices(int n, int k);

/**
 * @brief Prints a unified error message to stderr.
 * @return Always returns NULL.
 */
void* error(void) {
    fprintf(stderr, "An Error Has Occurred\n");
    return NULL;
}

/**
 * @brief Frees a 2D matrix of doubles. Safe to call on NULL.
 * @param m The matrix to free.
 * @param rows The number of rows in the matrix.
 */
void free_matrix(double **m, int rows) {
    int i;
    if (!m) return;
    for (i = 0; i < rows; i++) free(m[i]);
    free(m);
}

/**
 * @brief Allocates a 2D matrix of doubles, initialized to zero.
 * @param rows The number of rows to allocate.
 * @param cols The number of columns to allocate.
 * @return A pointer to the allocated matrix, or NULL on failure.
 */
static double** alloc_mat(int rows, int cols) {
    int i;
    double **M = (double**)malloc(rows * sizeof(double*));
    if (!M) return error();
    for (i = 0; i < rows; i++) {
        M[i] = (double*)calloc(cols, sizeof(double));
        if (!M[i]) { free_matrix(M, i); return error(); }
    }
    return M;
}

/**
 * @brief Calculates the squared Euclidean distance between two vectors.
 * @param a The first vector.
 * @param b The second vector.
 * @param d The dimension of the vectors.
 * @return The squared Euclidean distance.
 */
static double l2sq(const double *a, const double *b, int d) {
    int i;
    double s = 0.0, t;
    for (i = 0; i < d; i++) { t = a[i] - b[i]; s += t * t; }
    return s;
}

/**
 * @brief Calculates the similarity between two vectors using a Gaussian kernel.
 */
static double sim(const double *a, const double *b, int d) {
    return exp(-0.5 * l2sq(a, b, d));
}

/**
 * @brief Performs matrix multiplication: C = A * B.
 * @param r Rows of A.
 * @param m Columns of A / Rows of B.
 * @param c Columns of B.
 * @param A Matrix A.
 * @param B Matrix B.
 * @param C The result matrix C.
 */
static void mm(int r, int m, int c, double **A, double **B, double **C) {
    int i, j, k;
    double aik;
    for(i=0; i<r; i++) for(j=0; j<c; j++) C[i][j] = 0.0;
    
    for (i = 0; i < r; i++) {
        for (k = 0; k < m; k++) {
            aik = A[i][k];
            for (j = 0; j < c; j++) C[i][j] += aik * B[k][j];
        }
    }
}

/**
 * @brief Transposes a matrix: AT = A^T.
 */
static void tr(int r, int c, double **A, double **AT) {
    int i, j;
    for (i = 0; i < r; i++) for (j = 0; j < c; j++) AT[j][i] = A[i][j];
}

/**
 * @brief Calculates the squared Frobenius norm of the difference between two matrices.
 */
static double f_diff_sq(double **A, double **B, int r, int c) {
    int i, j;
    double s = 0.0, d;
    for (i = 0; i < r; i++) for (j = 0; j < c; j++) { d = A[i][j] - B[i][j]; s += d * d; }
    return s;
}

/**
 * @brief Computes the similarity matrix A from data points X.
 * @return The similarity matrix A, or NULL on failure.
 */
double** sym(double **data, int n, int d) {
    int i, j;
    double **A = alloc_mat(n, n);
    if (!A) return NULL;
    for (i = 0; i < n; i++) for (j = 0; j < n; j++)
        A[i][j] = (i == j) ? 0.0 : sim(data[i], data[j], d);
    return A;
}

/**
 * @brief Computes the diagonal degree matrix D from the similarity matrix A.
 * @return The diagonal degree matrix D, or NULL on failure.
 */
double** ddg(double **A, int n) {
    int i, j;
    double **D = alloc_mat(n, n);
    if (!D) return NULL;
    for (i = 0; i < n; i++) {
        double s = 0.0;
        for (j = 0; j < n; j++) s += A[i][j];
        D[i][i] = s;
    }
    return D;
}

/**
 * @brief Computes the normalized similarity matrix W = D^(-1/2) * A * D^(-1/2).
 * @return The normalized similarity matrix W, or NULL on failure.
 */
double** norm(double **A, double **D, int n) {
    int i, j;
    double **W = alloc_mat(n, n);
    if (!W) return NULL;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            double di = D[i][i];
            double dj = D[j][j];
            if (di > 0.0 && dj > 0.0) {
                W[i][j] = A[i][j] / (sqrt(di) * sqrt(dj));
            }
        }
    }
    return W;
}

/**
 * @brief Copies matrix 'src' into 'dst'.
 */
static void copy_mat(double **dst, double **src, int r, int c) {
    int i;
    for (i = 0; i < r; i++) memcpy(dst[i], src[i], c * sizeof(double));
}

/**
 * @brief Performs a single multiplicative update step for the H matrix.
 *
 * @param H The current H matrix to be updated (n x k).
 * @param Hprev A matrix to store the state of H before the update (n x k).
 * @param ... other params are temporary matrices for calculations.
 */
static void update_step(double **W, double **H, double **Hprev, double **HT,
                        double **HHT, double **WH, double **HHTH, int n, int k) {
    int i, j;
    const double beta = 0.5;

    copy_mat(Hprev, H, n, k);
    tr(n, k, H, HT);
    mm(n, k, n, H, HT, HHT);
    mm(n, n, k, W, H, WH);
    mm(n, n, k, HHT, H, HHTH);
    
    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            double numer = WH[i][j];
            double denom = HHTH[i][j];
            if (denom > 1e-12) {
                 H[i][j] = Hprev[i][j] * (1 - beta + beta * (numer / denom));
            }
        }
    }
}

/**
 * @brief Allocates a struct containing all temporary matrices for symNMF.
 *
 * @param n The number of data points (rows).
 * @param k The number of clusters (columns for H).
 * @return A pointer to the allocated SymnmfMatrices struct, or NULL on failure.
 */
static SymnmfMatrices* alloc_symnmf_matrices(int n, int k) {
    SymnmfMatrices *mats = malloc(sizeof(SymnmfMatrices));
    if (!mats) { error(); return NULL; }

    mats->H = alloc_mat(n, k);
    mats->Hprev = alloc_mat(n, k);
    mats->HT = alloc_mat(k, n);
    mats->HHT = alloc_mat(n, n);
    mats->WH = alloc_mat(n, k);
    mats->HHTH = alloc_mat(n, k);

    if (!mats->H || !mats->Hprev || !mats->HT || !mats->HHT || !mats->WH || !mats->HHTH) {
        free_matrix(mats->H, n);
        free_matrix(mats->Hprev, n);
        free_matrix(mats->HT, k);
        free_matrix(mats->HHT, n);
        free_matrix(mats->WH, n);
        free_matrix(mats->HHTH, n);
        free(mats);
        return NULL;
    }
    return mats;
}

/**
 * @brief Performs the full Symmetric Non-negative Matrix Factorization algorithm.
 *
 * @param W The normalized similarity matrix (n x n).
 * @param H0 The initial H matrix (n x k).
 * @param maxIter The maximum number of iterations.
 * @param eps The convergence threshold.
 * @return The final optimized H matrix (n x k). The caller must free it.
 */
double** symnmf(double **W, double **H0, int n, int k, int maxIter, double eps) {
    int t;
    double eps_sq = eps * eps;
    double **result_H;
    SymnmfMatrices *mats = alloc_symnmf_matrices(n, k);

    if (!mats) return NULL;
    
    copy_mat(mats->H, H0, n, k);
    
    for (t = 0; t < maxIter; t++) {
        update_step(W, mats->H, mats->Hprev, mats->HT, mats->HHT,
                    mats->WH, mats->HHTH, n, k);
        if (f_diff_sq(mats->H, mats->Hprev, n, k) < eps_sq) {
            break;
        }
    }
    
    result_H = mats->H;
    free_matrix(mats->Hprev, n);
    free_matrix(mats->HT, k);
    free_matrix(mats->HHT, n);
    free_matrix(mats->WH, n);
    free_matrix(mats->HHTH, n);
    free(mats);
    
    return result_H;
}

/*
 * Main function and file I/O for standalone execution
 */

/**
 * @brief Reads a single line from a file, dynamically allocating memory.
 */
static int read_line(FILE *fp, char **buf, size_t *cap) {
    size_t len = 0; int ch;
    if (!*buf || *cap == 0) {*cap=128; *buf=malloc(*cap); if(!*buf)return -1;}
    while ((ch = fgetc(fp)) != EOF && ch != '\n') {
        if (len + 1 >= *cap) {
            size_t newcap = *cap * 2;
            char *tmp = realloc(*buf, newcap);
            if (!tmp) return -1;
            *buf = tmp; *cap = newcap;
        }
        (*buf)[len++] = ch;
    }
    (*buf)[len] = '\0';
    return (len == 0 && ch == EOF) ? -1 : (int)len;
}

/**
 * @brief Reads a file once to determine its dimensions (rows, cols).
 */
static int count_shape(FILE *fp, int *outRows, int *outCols) {
    char *line = NULL; size_t cap = 0; int len, rows = 0, cols = 0, i;
    if ((len = read_line(fp, &line, &cap)) < 0) { free(line); return 0; }
    cols = 1; for (i = 0; i < len; i++) if (line[i] == ',') cols++;
    rows = 1;
    while ((len = read_line(fp, &line, &cap)) >= 0) if (len > 0) rows++;
    free(line);
    if (fseek(fp, 0L, SEEK_SET) != 0) return 0;
    *outRows = rows; *outCols = cols;
    return 1;
}

/**
 * @brief Parses a single line of comma-separated doubles.
 */
static int parse_row(char *line, int cols, double *out) {
    int j; char *p = line, *end;
    for (j = 0; j < cols; j++) {
        out[j] = strtod(p, &end);
        if (end == p) return 0;
        p = (*end == ',') ? end + 1 : end;
    }
    return 1;
}

/**
 * @brief Loads a CSV file into a 2D matrix of doubles.
 */
static double** load_csv(const char *path, int *outRows, int *outCols) {
    FILE *fp = fopen(path, "r");
    char *line = NULL; size_t cap = 0;
    int len, rows, cols, i;
    double **X;

    if (!fp) return error();
    if (!count_shape(fp, &rows, &cols)) { fclose(fp); return error(); }
    X = alloc_mat(rows, cols);
    if (!X) { fclose(fp); return NULL; }

    for (i = 0; i < rows; i++) {
        len = read_line(fp, &line, &cap);
        if (len < 0 || !parse_row(line, cols, X[i])) {
            free(line); free_matrix(X, rows); fclose(fp); return error();
        }
    }
    free(line); fclose(fp);
    *outRows = rows; *outCols = cols;
    return X;
}

/**
 * @brief Prints a 2D matrix to stdout, formatted to 4 decimal places.
 */
static void print_matrix(double **M, int rows, int cols) {
    int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            printf("%.4f%s", M[i][j], (j + 1 < cols) ? "," : "");
        }
        putchar('\n');
    }
}

/**
 * @brief Parses the goal string from command-line arguments.
 */
static int parse_goal(const char *g) {
    if (strcmp(g, "sym")  == 0) return 0;
    if (strcmp(g, "ddg")  == 0) return 1;
    if (strcmp(g, "norm") == 0) return 2;
    return -1;
}

/**
 * @brief Main entry point for the standalone C executable.
 *
 * @param argc The number of command-line arguments.
 * @param argv An array of command-line argument strings.
 * @return 0 on success, 1 on failure.
 */
int main(int argc, char **argv) {
    int goal, n, d;
    double **X, **A, **D, **W;
    
    if (argc != 3) { error(); return 1; }
    goal = parse_goal(argv[1]);
    if (goal == -1) { error(); return 1; }

    X = load_csv(argv[2], &n, &d);
    if (!X) return 1;

    A = sym(X, n, d);
    if (!A) { free_matrix(X, n); return 1; }
    if (goal == 0) { print_matrix(A, n, n); }

    if (goal >= 1) {
        D = ddg(A, n);
        if (!D) { free_matrix(A, n); free_matrix(X, n); return 1; }
        if (goal == 1) { print_matrix(D, n, n); }
        
        if (goal == 2) {
            W = norm(A, D, n);
            if (!W) { free_matrix(D,n); free_matrix(A,n); free_matrix(X,n); return 1; }
            print_matrix(W, n, n);
            free_matrix(W, n);
        }
        free_matrix(D, n);
    }

    free_matrix(A, n);
    free_matrix(X, n);
    return 0;
}