#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "symnmf.h"  /* error(), free_matrix(), sym(), ddg(), norm(), symnmf() */

/* ============================= Helper functions ============================= */

/* Print unified C error and set a Python exception; returns NULL for chaining. */
static void* py_error(void) {
    error();
    PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
    return NULL;
}

/* Allocate C matrix rows×cols (double**). Frees partial on failure. */
static double** alloc_c_matrix(int rows, int cols){
    int i;
    double **m = (double**)malloc(rows*sizeof(double*));
    if(!m) return py_error();
    for(i=0;i<rows;i++){
        m[i]=(double*)malloc(cols*sizeof(double));
        if(!m[i]){ free_matrix(m,i); return py_error(); }
    }
    return m;
}

/* Return 1 if Python object is int or float. */
static int is_number(PyObject *o){
    return PyFloat_Check(o) || PyLong_Check(o);
}

/* Validate list-of-lists of numbers; output shape in outRows/outCols. */
static int extract_dimensions(PyObject *py, int *outRows, int *outCols){
    Py_ssize_t rows, cols;
    PyObject *row0, *r, *cell;
    int i, j;
    if(!PyList_Check(py)) return (py_error(), 0);
    rows=PyList_Size(py); if(rows<=0) return (py_error(), 0);
    row0=PyList_GetItem(py,0); if(!PyList_Check(row0)) return (py_error(), 0);
    cols=PyList_Size(row0); if(cols<=0) return (py_error(), 0);
    for(i=0;i<(int)rows;i++){
        r=PyList_GetItem(py,i);
        if(!PyList_Check(r) || PyList_Size(r)!=cols) return (py_error(), 0);
        for(j=0;j<(int)cols;j++){ cell=PyList_GetItem(r,j); if(!is_number(cell)) return (py_error(), 0); }
    }
    *outRows=(int)rows; *outCols=(int)cols; return 1;
}

/* Convert Python list-of-lists -> C matrix; returns 1 on success. */
static int py_to_c_matrix(PyObject *py, double ***outM, int *outR, int *outC){
    int rows, cols, i, j;
    double **m;
    PyObject *row, *v;
    if(!extract_dimensions(py,&rows,&cols)) return 0;
    m=alloc_c_matrix(rows,cols); if(!m) return 0;
    for(i=0;i<rows;i++){
        row=PyList_GetItem(py,i);
        for(j=0;j<cols;j++){
            v=PyList_GetItem(row,j);
            m[i][j]= PyFloat_Check(v)? PyFloat_AsDouble(v):(double)PyLong_AsLongLong(v);
            if(PyErr_Occurred()){ free_matrix(m,rows); py_error(); return 0; }
        }
    }
    *outM=m; *outR=rows; *outC=cols; return 1;
}

/* Convert C matrix -> Python list-of-lists of floats. */
static PyObject* c_to_py_matrix(double **m, int rows, int cols){
    int i, j;
    PyObject *pyRows, *pyRow, *num;
    pyRows=PyList_New(rows); if(!pyRows) return py_error();
    for(i=0;i<rows;i++){
        pyRow=PyList_New(cols); if(!pyRow){ Py_DECREF(pyRows); return py_error(); }
        for(j=0;j<cols;j++){
            num=PyFloat_FromDouble(m[i][j]);
            if(!num){ Py_DECREF(pyRow); Py_DECREF(pyRows); return py_error(); }
            PyList_SET_ITEM(pyRow,j,num);
        }
        PyList_SET_ITEM(pyRows,i,pyRow);
    }
    return pyRows;
}

/* ============================= Python wrappers ============================= */

/* sym(data) -> similarity matrix */
static PyObject* py_sym(PyObject *self, PyObject *args){
    PyObject *pyData, *pyRes;
    double **data, **A;
    int n, d;
    pyData=NULL; pyRes=NULL; data=NULL; A=NULL; n=0; d=0;
    if(!PyArg_ParseTuple(args,"O",&pyData)) return py_error();
    if(!py_to_c_matrix(pyData,&data,&n,&d)) return NULL;
    A=sym(data,n,d); if(!A){ free_matrix(data,n); return py_error(); }
    pyRes=c_to_py_matrix(A,n,n);
    if(!pyRes){ free_matrix(A,n); free_matrix(data,n); return NULL; }
    free_matrix(A,n); free_matrix(data,n); return pyRes;
}

/* ddg(data) -> diagonal degree matrix */
static PyObject* py_ddg(PyObject *self, PyObject *args){
    PyObject *pyData, *pyRes;
    double **data, **A, **D;
    int n, d;
    pyData=NULL; pyRes=NULL; data=NULL; A=NULL; D=NULL; n=0; d=0;
    if(!PyArg_ParseTuple(args,"O",&pyData)) return py_error();
    if(!py_to_c_matrix(pyData,&data,&n,&d)) return NULL;
    A=sym(data,n,d); if(!A){ free_matrix(data,n); return py_error(); }
    D=ddg(A,n); if(!D){ free_matrix(A,n); free_matrix(data,n); return py_error(); }
    pyRes=c_to_py_matrix(D,n,n);
    if(!pyRes){ free_matrix(D,n); free_matrix(A,n); free_matrix(data,n); return NULL; }
    free_matrix(D,n); free_matrix(A,n); free_matrix(data,n); return pyRes;
}

/* norm(data) -> normalized similarity matrix */
static PyObject* py_norm(PyObject *self, PyObject *args){
    PyObject *pyData, *pyRes;
    double **data, **A, **D, **W;
    int n, d;
    pyData=NULL; pyRes=NULL; data=NULL; A=NULL; D=NULL; W=NULL; n=0; d=0;
    if(!PyArg_ParseTuple(args,"O",&pyData)) return py_error();
    if(!py_to_c_matrix(pyData,&data,&n,&d)) return NULL;
    A=sym(data,n,d); if(!A){ free_matrix(data,n); return py_error(); }
    D=ddg(A,n); if(!D){ free_matrix(A,n); free_matrix(data,n); return py_error(); }
    W=norm(A,D,n); if(!W){ free_matrix(D,n); free_matrix(A,n); free_matrix(data,n); return py_error(); }
    pyRes=c_to_py_matrix(W,n,n);
    if(!pyRes){ free_matrix(W,n); free_matrix(D,n); free_matrix(A,n); free_matrix(data,n); return NULL; }
    free_matrix(W,n); free_matrix(D,n); free_matrix(A,n); free_matrix(data,n); return pyRes;
}

/* symnmf(W_norm,H_init,max_iter,eps) -> factor matrix H */
static PyObject* py_symnmf(PyObject *self, PyObject *args){
    PyObject *pyW, *pyH0, *pyRes;
    double **W, **H0, **H;
    int nWr, nWc, nHr, nHc, n, k, maxIter;
    double eps;
    pyW=NULL; pyH0=NULL; pyRes=NULL; W=NULL; H0=NULL; H=NULL;
    nWr=0; nWc=0; nHr=0; nHc=0; n=0; k=0; maxIter=0; eps=0.0;
    if(!PyArg_ParseTuple(args,"OOid",&pyW,&pyH0,&maxIter,&eps)) return py_error();
    if(maxIter<=0 || eps<=0.0) return py_error();
    if(!py_to_c_matrix(pyW,&W,&nWr,&nWc)) return NULL;
    if(!py_to_c_matrix(pyH0,&H0,&nHr,&nHc)){ free_matrix(W,nWr); return NULL; }
    if(nWr!=nWc || nWr!=nHr || nHc<=0){ free_matrix(H0,nHr); free_matrix(W,nWr); return py_error(); }
    n=nWr; k=nHc;
    H=symnmf(W,H0,n,k,maxIter,eps);
    if(!H){ free_matrix(H0,nHr); free_matrix(W,n); return py_error(); }
    pyRes=c_to_py_matrix(H,n,k);
    if(!pyRes){ free_matrix(H,n); free_matrix(H0,nHr); free_matrix(W,n); return NULL; }
    free_matrix(H,n); free_matrix(H0,nHr); free_matrix(W,n); return pyRes;
}

/* ============================= Module initialization ============================= */

static PyMethodDef SymNMFMethods[] = {
    {"sym",    (PyCFunction)py_sym,    METH_VARARGS, "sym(data) -> similarity matrix"},
    {"ddg",    (PyCFunction)py_ddg,    METH_VARARGS, "ddg(data) -> diagonal degree matrix"},
    {"norm",   (PyCFunction)py_norm,   METH_VARARGS, "norm(data) -> normalized similarity matrix"},
    {"symnmf", (PyCFunction)py_symnmf, METH_VARARGS, "symnmf(W_norm,H_init,max_iter,eps) -> factor matrix H"},
    {NULL,NULL,0,NULL}
};

static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT, "symnmf", "C extension for SymNMF", -1, SymNMFMethods
};

PyMODINIT_FUNC PyInit_symnmf(void){
    return PyModule_Create(&symnmfmodule);
}
