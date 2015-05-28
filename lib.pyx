from numpy cimport *
cimport numpy as np
import numpy as np

cimport cython

import_array()

cdef float64_t FP_ERR = 1e-13

from libc.stdlib cimport malloc, free

from numpy cimport NPY_INT8 as NPY_int8
from numpy cimport NPY_INT16 as NPY_int16
from numpy cimport NPY_INT32 as NPY_int32
from numpy cimport NPY_INT64 as NPY_int64
from numpy cimport NPY_FLOAT16 as NPY_float16
from numpy cimport NPY_FLOAT32 as NPY_float32
from numpy cimport NPY_FLOAT64 as NPY_float64

from numpy cimport (int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t,
                    uint32_t, uint64_t, float32_t, float64_t)

int8 = np.dtype(np.int8)
int16 = np.dtype(np.int16)
int32 = np.dtype(np.int32)
int64 = np.dtype(np.int64)
float16 = np.dtype(np.float16)
float32 = np.dtype(np.float32)
float64 = np.dtype(np.float64)

cdef np.int8_t MINint8 = np.iinfo(np.int8).min
cdef np.int16_t MINint16 = np.iinfo(np.int16).min
cdef np.int32_t MINint32 = np.iinfo(np.int32).min
cdef np.int64_t MINint64 = np.iinfo(np.int64).min
cdef np.float32_t MINfloat32 = np.NINF
cdef np.float64_t MINfloat64 = np.NINF

cdef np.int8_t MAXint8 = np.iinfo(np.int8).max
cdef np.int16_t MAXint16 = np.iinfo(np.int16).max
cdef np.int32_t MAXint32 = np.iinfo(np.int32).max
cdef np.int64_t MAXint64 = np.iinfo(np.int64).max
cdef np.float32_t MAXfloat32 = np.inf
cdef np.float64_t MAXfloat64 = np.inf

cdef double NaN = <double> np.NaN
cdef double nan = NaN

cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b

_return_false = lambda self, other: False
_return_true = lambda self, other: True

ctypedef fused numeric:
    int8_t
    int16_t
    int32_t
    int64_t

    uint8_t
    uint16_t
    uint32_t
    uint64_t

    float32_t
    float64_t

cdef inline Py_ssize_t swap(numeric *a, numeric *b) except -1:
    cdef numeric t

    # cython doesn't allow pointer dereference so use array syntax
    t = a[0]
    a[0] = b[0]
    b[0] = t
    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef numeric kth_smallest(numeric[:] a, Py_ssize_t k):
    cdef:
        Py_ssize_t i, j, l, m, n = a.size
        numeric x

    l = 0
    m = n - 1

    while l < m:
        x = a[k]
        i = l
        j = m

        while 1:
            while a[i] < x: i += 1
            while x < a[j]: j -= 1
            if i <= j:
                swap(&a[i], &a[j])
                i += 1; j -= 1

            if i > j: break

        if j < k: l = i
        if k < i: m = j
    return a[k]