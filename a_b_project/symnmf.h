#ifndef SYMNMF_H_
#define SYMNMF_H_

/*
 * This header defines the function prototypes for the core C implementation
 * of the SymNMF algorithm. These functions are used by both the C executable
 * (symnmf.c) and the Python C-API wrapper (symnmfmodule.c).
 */

/* Function Prototypes */
double** sym(double **data, int n, int d);
double** ddg(double **A, int n);
double** norm(double **A, double **D, int n);
double** symnmf(double **W, double **H0, int n, int k, int maxIter, double eps);

/* Utility Functions */
void free_matrix(double **matrix, int num_rows);
void* error(void);

#endif