#ifndef SYMNMF_H_
#define SYMNMF_H_

/**
 * @file symnmf.h
 * @brief Header for the C implementation of the SymNMF algorithm.
 * * Defines function prototypes used by both the C executable (symnmf.c)
 * and the Python C-API wrapper (symnmfmodule.c).
 */

/* ============================ Core API Functions ============================ */

/**
 * @brief Computes the similarity matrix A from data points X.
 * @param data The input data matrix (n x d).
 * @param n The number of data points.
 * @param d The dimension of each data point.
 * @return The (n x n) similarity matrix A, or NULL on failure.
 */
double** sym(double **data, int n, int d);

/**
 * @brief Computes the diagonal degree matrix D from the similarity matrix A.
 * @param A The (n x n) similarity matrix.
 * @param n The number of data points.
 * @return The (n x n) diagonal degree matrix D, or NULL on failure.
 */
double** ddg(double **A, int n);

/**
 * @brief Computes the normalized similarity matrix W.
 * @param A The (n x n) similarity matrix.
 * @param D The (n x n) diagonal degree matrix.
 * @param n The number of data points.
 * @return The (n x n) normalized similarity matrix W, or NULL on failure.
 */
double** norm(double **A, double **D, int n);

/**
 * @brief Performs the full SymNMF optimization to find H.
 * @param W The (n x n) normalized similarity matrix.
 * @param H0 The initial (n x k) H matrix.
 * @param n The number of data points.
 * @param k The number of clusters.
 * @param maxIter The maximum number of iterations.
 * @param eps The convergence threshold.
 * @return The final optimized (n x k) H matrix, or NULL on failure.
 */
double** symnmf(double **W, double **H0, int n, int k, int maxIter, double eps);


/* =========================== Utility Functions ============================ */

/**
 * @brief Frees a dynamically allocated 2D matrix.
 * @param matrix The matrix to free.
 * @param num_rows The number of rows in the matrix.
 */
void free_matrix(double **matrix, int num_rows);

/**
 * @brief Prints a unified error message to stderr.
 * @return Always returns NULL, suitable for chained returns.
 */
void* error(void);

#endif /* SYMNMF_H_ */
