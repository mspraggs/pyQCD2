"""
This module handles all linear algebra relating to the combination of field.
"""

from __future__ import absolute_import

cimport cython

import numpy as np
cimport numpy as cnp


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def mult_gauge_gauge(cnp.ndarray[cnp.complex128_t, ndim=3] U1,
                     cnp.ndarray[cnp.complex128_t, ndim=3] U2):
    """Takes two gauge fields and multiplies their SU(N) matrices together."""
    cdef int i, j, a, b, c, N, nc
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] out
    N = U1.shape[0]
    nc = U2.shape[1]
    out = np.zeros((N, nc, nc), dtype=np.complex)

    for i in range(N):
        for a in range(nc):
            for b in range(nc):
                for c in range(nc):
                    out[i, a, c] = out[i, a, c] + U1[i, a, b] * U2[i, b, c]

    return out
