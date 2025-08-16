#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"

/* ============================= Helper Functions =============================
 * py_error
 * Purpose: set a Python RuntimeError with the unified message.
 * Returns: NULL (so callers can `return py_error();` in error paths).
 */
static PyObject* py_error(void) {
    PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
    return NULL;
}

/* py_to_c_matrix
 * Purpose: Convert a Python list-of-lists (rows) into a newly malloc'ed
 *          C double** matrix; validates rectangularity.
 * On success: writes matrix pointer, rows, cols into out-params and returns 1.
 * On failure: frees any partial allocation and returns 0 (no Python error set).
 * Ownership: caller must free with free_matrix(c_matrix, rows).
 */
static int py_to_c_matrix(PyObject *py_matrix, double ***c_matrix_ptr, int *rows_ptr, int *cols_ptr) {
    Py_ssize_t i, j, rows, cols;
    double **c_matrix;
    PyObject *row_obj, *cell_obj;
    if (!PyList_Check(py_matrix) || (rows = PyList_Size(py_matrix)) == 0) return 0;
    row_obj = PyList_GetItem(py_matrix, 0);
    if (!PyList_Check(row_obj) || (cols = PyList_Size(row_obj)) == 0) return 0;
    c_matrix = (double**)malloc(rows * sizeof(double*));
    if (!c_matrix) return 0;
    for (i = 0; i < rows; i++) {
        row_obj = PyList_GetItem(py_matrix, i);
        if (!PyList_Check(row_obj) || PyList_Size(row_obj) != cols) {
            free_matrix(c_matrix, i); return 0;
        }
        c_matrix[i] = (double*)malloc(cols * sizeof(double));
        if (!c_matrix[i]) { free_matrix(c_matrix, i); return 0; }
        for (j = 0; j < cols; j++) {
            cell_obj = PyList_GetItem(row_obj, j);
            c_matrix[i][j] = PyFloat_AsDouble(cell_obj);
        }
    }
    *c_matrix_ptr = c_matrix;
    *rows_ptr = rows;
    *cols_ptr = cols;
    return 1;
}

/* c_to_py_matrix
 * Purpose: Convert a C double** matrix (rows×cols) into a Python list-of-lists.
 * Errors: returns NULL (with Python exception set) if allocation fails.
 * Ownership: caller keeps/frees the C matrix; this function only builds Python.
 */
static PyObject* c_to_py_matrix(double **c_matrix, int rows, int cols) {
    Py_ssize_t i, j;
    PyObject *py_list = PyList_New(rows);
    if (!py_list) return NULL;
    for (i = 0; i < rows; i++) {
        PyObject *py_row = PyList_New(cols);
        if (!py_row) { Py_DECREF(py_list); return NULL; }
        for (j = 0; j < cols; j++) {
            PyList_SET_ITEM(py_row, j, PyFloat_FromDouble(c_matrix[i][j]));
        }
        PyList_SET_ITEM(py_list, i, py_row);
    }
    return py_list;
}

/* ========================== Python Wrapper Functions ========================
 * py_sym
 * Args: data (list[list[float]]) with shape n×d.
 * Does: calls C sym() to compute A (n×n); converts to Python; frees C buffers.
 * Returns: Python list-of-lists A, or raises RuntimeError on failure.
 */
static PyObject* py_sym(PyObject *self, PyObject *args) {
    PyObject *py_data, *py_res;
    double **data, **A;
    int n, d;
    if (!PyArg_ParseTuple(args, "O", &py_data)) return py_error();
    if (!py_to_c_matrix(py_data, &data, &n, &d)) return py_error();
    A = sym(data, n, d);
    free_matrix(data, n);
    if (!A) return py_error();
    py_res = c_to_py_matrix(A, n, n);
    free_matrix(A, n);
    return py_res;
}

/* py_ddg
 * Args: A (n×n) similarity matrix as list-of-lists.
 * Does: validates square; calls C ddg(A); returns D (n×n) as Python.
 * Errors: raises RuntimeError on invalid shape or allocation failure.
 */
static PyObject* py_ddg(PyObject *self, PyObject *args) {
    PyObject *py_A, *py_res;
    double **A = NULL, **D;
    int n, n_cols;
    if (!PyArg_ParseTuple(args, "O", &py_A)) return py_error();
    if (!py_to_c_matrix(py_A, &A, &n, &n_cols) || n != n_cols) {
        if (A) free_matrix(A, n);
        return py_error();
    }
    D = ddg(A, n);
    free_matrix(A, n);
    if (!D) return py_error();
    py_res = c_to_py_matrix(D, n, n);
    free_matrix(D, n);
    return py_res;
}

/* py_norm
 * Args: A (n×n), D (n×n) as Python matrices.
 * Does: validates both are square and same n; calls C norm(A,D); returns W.
 * Errors: raises RuntimeError on invalid inputs or allocation failure.
 */
static PyObject* py_norm(PyObject *self, PyObject *args) {
    PyObject *py_A, *py_D, *py_res;
    double **A = NULL, **D = NULL, **W;
    int nA, nA_cols, nD, nD_cols;
    if (!PyArg_ParseTuple(args, "OO", &py_A, &py_D)) return py_error();
    if (!py_to_c_matrix(py_A, &A, &nA, &nA_cols) || nA != nA_cols) {
        if (A) free_matrix(A, nA); 
        return py_error();
    }
    if (!py_to_c_matrix(py_D, &D, &nD, &nD_cols) || nD != nD_cols || nA != nD) {
        free_matrix(A, nA);
        if (D) free_matrix(D, nD);
        return py_error();
    }
    W = norm(A, D, nA);
    free_matrix(A, nA);
    free_matrix(D, nD);
    if (!W) return py_error();
    py_res = c_to_py_matrix(W, nA, nA);
    free_matrix(W, nA);
    return py_res;
}

/* py_symnmf
 * Args: W (n×n), H0 (n×k), maxIter (int), eps (float).
 * Does: shape-checks; calls C symnmf(W,H0,n,k,maxIter,eps); returns H (n×k).
 * Errors: raises RuntimeError on invalid shapes or allocation failure.
 */
static PyObject* py_symnmf(PyObject *self, PyObject *args) {
    PyObject *py_W, *py_H0, *py_res;
    double **W = NULL, **H0 = NULL, **H;
    int n, k, maxIter, n_W_rows, n_W_cols, n_H0_rows;
    double eps;
    if (!PyArg_ParseTuple(args, "OOid", &py_W, &py_H0, &maxIter, &eps)) return py_error();
    if (!py_to_c_matrix(py_W, &W, &n_W_rows, &n_W_cols) || n_W_rows != n_W_cols) {
        if (W) free_matrix(W, n_W_rows);
        return py_error();
    }
    if (!py_to_c_matrix(py_H0, &H0, &n_H0_rows, &k)) {
        free_matrix(W, n_W_rows);
        if (H0) free_matrix(H0, n_H0_rows);
        return py_error();
    }
    n = n_W_rows;
    if (n != n_H0_rows) {
        free_matrix(W, n);
        free_matrix(H0, n_H0_rows);
        return py_error();
    }
    H = symnmf(W, H0, n, k, maxIter, eps);
    free_matrix(W, n);
    free_matrix(H0, n_H0_rows);
    if (!H) return py_error();
    py_res = c_to_py_matrix(H, n, k);
    free_matrix(H, n);
    return py_res;
}

/* ========================== Module Registration ===========================
 * Method table: maps Python names to C functions; docstrings are brief.
 */
static PyMethodDef SymNMFMethods[] = {
    {"sym", (PyCFunction)py_sym, METH_VARARGS, "Calculate similarity matrix A."},
    {"ddg", (PyCFunction)py_ddg, METH_VARARGS, "Calculate diagonal degree matrix D."},
    {"norm", (PyCFunction)py_norm, METH_VARARGS, "Calculate normalized similarity matrix W."},
    {"symnmf", (PyCFunction)py_symnmf, METH_VARARGS, "Run SymNMF optimization."},
    {NULL, NULL, 0, NULL}
};

/* Module definition: name 'symnmf', brief module doc, no per-interpreter state. */
static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT, "symnmf", "C extension for SymNMF", -1, SymNMFMethods
};

/* PyInit_symnmf
 * Purpose: module initialization entry point for Python.
 * Returns: new module object, or NULL on failure.
 */
PyMODINIT_FUNC PyInit_symnmf(void) {
    return PyModule_Create(&symnmfmodule);
}
