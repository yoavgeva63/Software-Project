#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "symnmf.h"

/* Print unified error; returns NULL so callers can `return error();`. */
void* error(void) {
    fprintf(stderr, "An Error Has Occurred\n");
    return NULL;
}

/* Free a rows×? matrix (NULL-safe). */
void free_matrix(double **m, int rows) {
    int i;
    if (!m) return;
    for (i = 0; i < rows; i++) free(m[i]);
    free(m);
}

/* Allocate rows×cols double matrix; on failure prints error and returns NULL. */
static double** alloc_mat(int rows, int cols) {
    int i;
    double **M = (double**)malloc(rows * sizeof(double*));
    if (!M) return error();
    for (i = 0; i < rows; i++) {
        M[i] = (double*)malloc(cols * sizeof(double));
        if (!M[i]) { free_matrix(M, i); return error(); }
    }
    return M;
}

/* Squared Euclidean distance between vectors a,b of dimension d. */
static double l2sq(const double *a, const double *b, int d) {
    int i;
    double s = 0.0, t;
    for (i = 0; i < d; i++) { t = a[i] - b[i]; s += t * t; }
    return s;
}

/* Gaussian-like similarity: exp(-0.5 * ||a-b||^2). */
static double sim(const double *a, const double *b, int d) {
    return exp(-0.5 * l2sq(a, b, d));
}

/* Set all entries of C[r×c] to zero. */
static void zero(double **C, int r, int c) {
    int i, j;
    for (i = 0; i < r; i++) for (j = 0; j < c; j++) C[i][j] = 0.0;
}

/* Matrix multiply: C = A[r×m] * B[m×c]. */
static void mm(int r, int m, int c, double **A, double **B, double **C) {
    int i, j, k; double aik;
    zero(C, r, c);
    for (i = 0; i < r; i++)
        for (k = 0; k < m; k++) {
            aik = A[i][k];
            for (j = 0; j < c; j++) C[i][j] += aik * B[k][j];
        }
}

/* Transpose: AT = A[r×c]^T (AT is c×r). */
static void tr(int r, int c, double **A, double **AT) {
    int i, j;
    for (i = 0; i < r; i++) for (j = 0; j < c; j++) AT[j][i] = A[i][j];
}

/* Frobenius norm of (A - B). */
static double f_diff(double **A, double **B, int r, int c) {
    int i, j; double s = 0.0, d;
    for (i = 0; i < r; i++) for (j = 0; j < c; j++) { d = A[i][j] - B[i][j]; s += d * d; }
    return sqrt(s);
}

/* Build similarity matrix A (n×n) from data (n×d); zero diagonal. */
double** sym(double **data, int n, int d) {
    int i, j;
    double **A = alloc_mat(n, n); if (!A) return NULL;
    for (i = 0; i < n; i++) for (j = 0; j < n; j++)
        A[i][j] = (i == j) ? 0.0 : sim(data[i], data[j], d);
    return A;
}

/* Build diagonal degree matrix D (n×n) from similarity A. */
double** ddg(double **A, int n) {
    int i, j; double **D = alloc_mat(n, n); double s;
    if (!D) return NULL;
    for (i = 0; i < n; i++) {
        s = 0.0;
        for (j = 0; j < n; j++) s += A[i][j];
        for (j = 0; j < n; j++) D[i][j] = 0.0;
        D[i][i] = s;
    }
    return D;
}

/* Normalized similarity: W = D^{-1/2} A D^{-1/2}. */
double** norm(double **A, double **D, int n) {
    int i, j; double **W = alloc_mat(n, n); double di, dj, dis, djs;
    if (!W) return NULL;
    for (i = 0; i < n; i++) {
        di = D[i][i]; dis = (di > 0.0) ? 1.0 / sqrt(di) : 0.0;
        for (j = 0; j < n; j++) {
            dj = D[j][j]; djs = (dj > 0.0) ? 1.0 / sqrt(dj) : 0.0;
            W[i][j] = A[i][j] * dis * djs;
        }
    }
    return W;
}

/* Copy matrix src[r×c] into dst[r×c]. */
static void copy_mat(double **dst, double **src, int r, int c) {
    int i, j;
    for (i = 0; i < r; i++) for (j = 0; j < c; j++) dst[i][j] = src[i][j];
}

/* One SymNMF multiplicative-update step; updates H in-place. */
static void update_step(
    double **W, double **H, double **Hprev, double **HT,
    double **HHT, double **WH, double **HHTH, int n, int k
) {
    int i, j;
    copy_mat(Hprev, H, n, k);
    tr(n, k, H, HT);
    mm(n, k, n, H,  HT,  HHT);   /* H H^T   */
    mm(n, n, k, HHT, H,  HHTH);  /* H H^T H */
    mm(n, n, k, W,   H,  WH);    /* W H     */
    for (i = 0; i < n; i++) for (j = 0; j < k; j++) {
        double denom = HHTH[i][j] + 1e-15;
        double v = H[i][j] * (WH[i][j] / denom);
        H[i][j] = (v > 0.0) ? v : 0.0;
    }
}

/* Symmetric NMF (returns new H; caller frees). */
double** symnmf(double **W, double **H0, int n, int k, int maxIter, double eps) {
    int t;
    double **H = alloc_mat(n, k), **Hprev = alloc_mat(n, k);
    double **HT = alloc_mat(k, n), **HHT = alloc_mat(n, n);
    double **WH = alloc_mat(n, k), **HHTH = alloc_mat(n, k);
    if (!H || !Hprev || !HT || !HHT || !WH || !HHTH) {
        if (HHTH) free_matrix(HHTH,n); if (WH) free_matrix(WH,n);
        if (HHT) free_matrix(HHT,n);   if (HT) free_matrix(HT,k);
        if (Hprev) free_matrix(Hprev,n); if (H) free_matrix(H,n);
        return NULL;
    }
    copy_mat(H, H0, n, k);
    for (t = 0; t < maxIter; t++) {
        update_step(W,H,Hprev,HT,HHT,WH,HHTH,n,k);
        if (f_diff(H, Hprev, n, k) < eps) break;
    }
    free_matrix(HHTH,n); free_matrix(WH,n); free_matrix(HHT,n);
    free_matrix(HT,k);   free_matrix(Hprev,n);
    return H;
}

/* Read a single line (without newline) via fgetc/realloc; returns length or -1 on EOF/error. */
static int read_line(FILE *fp, char **buf, size_t *cap) {
    size_t len = 0, newcap; int ch; char *tmp;
    if (!*buf || *cap == 0) { *cap = 128; *buf = (char*)malloc(*cap); if (!*buf) return -1; }
    while ((ch = fgetc(fp)) != EOF) {
        if (len + 1 >= *cap) { newcap = (*cap < 16384) ? (*cap * 2) : (*cap + 8192);
            tmp = (char*)realloc(*buf, newcap); if (!tmp) return -1; *buf = tmp; *cap = newcap; }
        if (ch == '\n') break; (*buf)[len++] = (char)ch;
    }
    if (len == 0 && ch == EOF) return -1; (*buf)[len] = '\0'; return (int)len;
}

/* Count rows/cols (by commas) and rewind file to start. */
static int count_shape(FILE *fp, int *outRows, int *outCols) {
    char *line = NULL; size_t cap = 0; int len, rows, cols, i;
    rows = 0; cols = 0;
    len = read_line(fp, &line, &cap); if (len < 0) { free(line); return 0; }
    cols = 1; for (i = 0; i < len; i++) if (line[i] == ',') cols++;
    rows = 1;
    while ((len = read_line(fp, &line, &cap)) >= 0) if (len > 0) rows++;
    free(line); if (fseek(fp, 0L, SEEK_SET) != 0) return 0;
    *outRows = rows; *outCols = cols; return 1;
}

/* Parse a CSV row with exactly cols values into out[]. */
static int parse_row(char *line, int cols, double *out) {
    int j; char *p = line, *end;
    for (j = 0; j < cols; j++) { out[j] = strtod(p, &end); if (end == p) return 0; p = (*end == ',') ? end + 1 : end; }
    return 1;
}

/* Load CSV at path into rows×cols matrix; returns NULL on error. */
static double** load_csv(const char *path, int *outRows, int *outCols) {
    FILE *fp; char *line; size_t cap; int len, rows, cols, i; double **X;
    fp = fopen(path, "r"); if (!fp) return error(); line = NULL; cap = 0;
    if (!count_shape(fp, &rows, &cols)) { fclose(fp); return error(); }
    X = alloc_mat(rows, cols); if (!X) { fclose(fp); return NULL; }
    for (i = 0; i < rows; i++) {
        len = read_line(fp, &line, &cap);
        if (len < 0 || !parse_row(line, cols, X[i])) { free(line); free_matrix(X, rows); fclose(fp); return error(); }
    }
    free(line); fclose(fp); *outRows = rows; *outCols = cols; return X;
}

/* Print rows×cols matrix with 4 decimal places, space-separated. */
static void print_matrix(double **M, int rows, int cols) {
    int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) printf("%.4f%s", M[i][j], (j + 1 < cols) ? " " : "");
        putchar('\n');
    }
}

/* Parse goal string -> 0:sym, 1:ddg, 2:norm, -1:invalid. */
static int parse_goal(const char *g) {
    if (strcmp(g, "sym")  == 0) return 0;
    if (strcmp(g, "ddg")  == 0) return 1;
    if (strcmp(g, "norm") == 0) return 2;
    return -1;
}

/* CLI: argv[1]=goal (sym|ddg|norm), argv[2]=csv path; prints result matrix. */
int main(int argc, char **argv) {
    int goal, n, d;
    double **X, **A, **D, **W;
    if (argc != 3) return (error(), 0);
    goal = parse_goal(argv[1]); if (goal == -1) return (error(), 0);

    X = load_csv(argv[2], &n, &d); if (!X) return 0;
    A = sym(X, n, d); if (!A) { free_matrix(X, n); return 0; }

    if (goal == 0) { print_matrix(A, n, n); free_matrix(A, n); free_matrix(X, n); return 0; }
    D = ddg(A, n); if (!D) { free_matrix(A, n); free_matrix(X, n); return 0; }
    if (goal == 1) { print_matrix(D, n, n); free_matrix(D, n); free_matrix(A, n); free_matrix(X, n); return 0; }
    W = norm(A, D, n); if (!W) { free_matrix(D, n); free_matrix(A, n); free_matrix(X, n); return 0; }
    print_matrix(W, n, n);
    free_matrix(W, n); free_matrix(D, n); free_matrix(A, n); free_matrix(X, n);
    return 0;
}
