#include "symnmf.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    double **H, **Hprev, **HT, **HHT, **WH, **HHTH;
} SymnmfMatrices;

/* Forward declarations for static helpers */
static void mm(int r, int m, int c, double **A, double **B, double **C);
static int run_goal(int goal, int n, int d, double **X);
static int parse_goal(const char *g_str);

void* error(void) {
    fprintf(stderr, "An Error Has Occurred\n");
    return NULL;
}

void free_matrix(double **m, int rows) {
    int i;
    if (!m) return;
    for (i = 0; i < rows; i++) free(m[i]);
    free(m);
}

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

static double l2sq(const double *a, const double *b, int d) {
    int i;
    double s = 0.0, t;
    for (i = 0; i < d; i++) { t = a[i] - b[i]; s += t * t; }
    return s;
}

static double sim(const double *a, const double *b, int d) {
    return exp(-0.5 * l2sq(a, b, d));
}

static void mm(int r, int m, int c, double **A, double **B, double **C) {
    int i, j, k;
    for (i = 0; i < r; i++) {
        for (j = 0; j < c; j++) {
            double sum = 0.0;
            for (k = 0; k < m; k++) sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
    }
}

static void tr(int r, int c, double **A, double **AT) {
    int i, j;
    for (i = 0; i < r; i++) for (j = 0; j < c; j++) AT[j][i] = A[i][j];
}

static double f_diff_sq(double **A, double **B, int r, int c) {
    int i, j;
    double s = 0.0, d;
    for (i = 0; i < r; i++) for (j = 0; j < c; j++) { d = A[i][j] - B[i][j]; s += d * d; }
    return s;
}

double** sym(double **data, int n, int d) {
    int i, j;
    double **A = alloc_mat(n, n);
    if (!A) return NULL;
    for (i = 0; i < n; i++) for (j = 0; j < n; j++)
        A[i][j] = (i == j) ? 0.0 : sim(data[i], data[j], d);
    return A;
}

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

double** norm(double **A, double **D, int n) {
    int i, j;
    double **W = alloc_mat(n, n);
    if (!W) return NULL;
    for (i = 0; i < n; i++) for (j = 0; j < n; j++) {
        double di = D[i][i], dj = D[j][j];
        if (di > 0.0 && dj > 0.0)
            W[i][j] = A[i][j] / (sqrt(di) * sqrt(dj));
    }
    return W;
}

static void copy_mat(double **dst, double **src, int r, int c) {
    int i;
    for (i = 0; i < r; i++) memcpy(dst[i], src[i], c * sizeof(double));
}

static void update_step(SymnmfMatrices *m, double **W, int n, int k) {
    const double beta = 0.5;
    int i, j;
    copy_mat(m->Hprev, m->H, n, k);
    tr(n, k, m->H, m->HT);
    mm(n, k, n, m->H, m->HT, m->HHT);
    mm(n, n, k, W, m->H, m->WH);
    mm(n, n, k, m->HHT, m->H, m->HHTH);
    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            double denom = m->HHTH[i][j];
            if (denom > 1e-12)
                m->H[i][j] = m->Hprev[i][j] * (1-beta+beta * (m->WH[i][j]/denom));
        }
    }
}

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
        free_matrix(mats->H, n); free_matrix(mats->Hprev, n);
        free_matrix(mats->HT, k); free_matrix(mats->HHT, n);
        free_matrix(mats->WH, n); free_matrix(mats->HHTH, n);
        free(mats); return NULL;
    }
    return mats;
}

double** symnmf(double **W, double **H0, int n, int k, int maxIter, double eps) {
    int t;
    double **result_H;
    SymnmfMatrices *mats = alloc_symnmf_matrices(n, k);
    if (!mats) return NULL;
    copy_mat(mats->H, H0, n, k);
    for (t = 0; t < maxIter; t++) {
        update_step(mats, W, n, k);
        if (f_diff_sq(mats->H, mats->Hprev, n, k) < eps) break;
    }
    result_H = mats->H;
    free_matrix(mats->Hprev, n); free_matrix(mats->HT, k);
    free_matrix(mats->HHT, n); free_matrix(mats->WH, n);
    free_matrix(mats->HHTH, n); free(mats);
    return result_H;
}

/* ================== Standalone C Executable Logic =================== */

static int read_line(FILE *fp, char **buf, size_t *cap) {
    size_t len = 0; int ch;
    if (!*buf || *cap == 0) {*cap=128; *buf=malloc(*cap); if(!*buf)return -1;}
    while ((ch = fgetc(fp)) != EOF && ch != '\n') {
        if (len + 1 >= *cap) {
            char *tmp = realloc(*buf, (*cap) *= 2);
            if (!tmp) return -1;
            *buf = tmp;
        }
        (*buf)[len++] = ch;
    }
    (*buf)[len] = '\0';
    return (len == 0 && ch == EOF) ? -1 : (int)len;
}

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

static double** load_csv(const char *path, int *outRows, int *outCols) {
    FILE *fp = fopen(path, "r");
    char *line = NULL; size_t cap = 0;
    int i, j;
    double **X;
    if (!fp) return error();
    if (!count_shape(fp, outRows, outCols)) { fclose(fp); return error(); }
    X = alloc_mat(*outRows, *outCols);
    if (!X) { fclose(fp); return NULL; }
    for (i = 0; i < *outRows; i++) {
        read_line(fp, &line, &cap);
        char *p = line, *end;
        for (j = 0; j < *outCols; j++) {
            X[i][j] = strtod(p, &end);
            p = (*end == ',') ? end + 1 : end;
        }
    }
    free(line); fclose(fp);
    return X;
}

static void print_matrix(double **M, int rows, int cols) {
    int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++)
            printf("%.4f%s", M[i][j], (j + 1 < cols) ? "," : "");
        putchar('\n');
    }
}

static int parse_goal(const char *g_str) {
    if (strcmp(g_str, "sym") == 0) return 0;
    if (strcmp(g_str, "ddg") == 0) return 1;
    if (strcmp(g_str, "norm") == 0) return 2;
    return -1;
}

static int run_goal(int goal, int n, double **A) {
    double **D, **W;
    if (goal == 0) { print_matrix(A, n, n); return 1; }
    D = ddg(A, n);
    if (!D) return 0;
    if (goal == 1) { print_matrix(D, n, n); free_matrix(D, n); return 1; }
    W = norm(A, D, n);
    free_matrix(D, n);
    if (!W) return 0;
    print_matrix(W, n, n);
    free_matrix(W, n);
    return 1;
}

int main(int argc, char **argv) {
    int goal, n = 0, d = 0, res = 1; /* FIX: Initialize n,d */
    double **X, **A;
    if (argc != 3) { error(); return 1; }
    goal = parse_goal(argv[1]);
    if (goal == -1) { error(); return 1; }
    X = load_csv(argv[2], &n, &d);
    if (!X) return 1;
    A = sym(X, n, d);
    if (!A) {
        res = 0;
    } else {
        if (!run_goal(goal, n, A)) res = 0;
        free_matrix(A, n);
    }
    free_matrix(X, n);
    return res ? 0 : 1;
}
