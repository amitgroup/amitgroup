
# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature=True
# cython: cdivision=True
# cython: profile=True
import cython
import numpy as np
cimport numpy as np
from math import fabs
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

def deform_x(np.ndarray[DTYPE_t, ndim=3] xs, np.ndarray[DTYPE_t, ndim=3] u, int n, np.ndarray[DTYPE_t, ndim=4] psis):
    cdef:
        np.ndarray[DTYPE_t, ndim=3] zs = np.copy(xs) 
        DTYPE_t[:,:,:] xs_mv = xs
        DTYPE_t[:,:,:] zs_mv = zs    
        DTYPE_t[:,:,:] u_mv = u
        DTYPE_t[:,:,:,:] psis_mv = psis
        int sx0 = xs.shape[0]
        int sx1 = xs.shape[1]
        DTYPE_t u0, u1
        DTYPE_t ps
        int x0, x1, k1, k2
    
    with nogil:
        for x0 in range(sx0):
            for x1 in range(sx1):
                u0 = 0.0
                u1 = 0.0
                for k1 in range(n):
                    for k2 in range(n):
                        ps = psis_mv[k1,k2,x0,x1]
                        u0 += u_mv[0,k1,k2] * ps
                        u1 += u_mv[1,k1,k2] * ps
                zs_mv[x0,x1,0] += u0 
                zs_mv[x0,x1,1] += u1 
    return zs

