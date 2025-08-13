#include <Python.h>

# ifndef SYMNMF_H_
# define SYMNMF_H_

double** sym (double **data, int vectorCount, int dimension);
double** ddg (double **similarityMatrix, int vectorCount);
double** norm(double **similarityMatrix, double **degreeMatrix, int vectorCount);
double** symnmf(double **W, double **H0, int n, int k, int maxIter, double eps);
void free_matrix(double** matrix, int num_rows);
void* error(void);

# endif
