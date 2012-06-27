
# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature=True
# cython: profile=True
# cython: cdivision=True
import cython
import numpy as np
cimport numpy as np
#from cython.parallel import prange
from math import fabs
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


cdef inline DTYPE_t dabs(DTYPE_t x) nogil: 
    return x if x >= 0 else -x 

cdef inline DTYPE_t lerp(DTYPE_t a, DTYPE_t x, DTYPE_t y) nogil:
    return (1.0-a) * x + a * y

def interp2d(np.ndarray[DTYPE_t, ndim=3] xs, z, dx=None, startx=None, fill_value=None): 
    assert(xs.dtype == DTYPE)
    assert(z.dtype == DTYPE)
    dx = dx if dx is not None else 1.0/np.array(z.shape) #np.array([xs[1,0,0]-xs[0,0,0], xs[0,1,1]-xs[0,0,1]])
    startx = startx if startx is not None else np.zeros(2) 

    cdef DTYPE_t startx0 = startx[0]
    cdef DTYPE_t startx1 = startx[1]
    cdef DTYPE_t dx0 = dx[0]
    cdef DTYPE_t dx1 = dx[1]

    cdef np.ndarray[DTYPE_t, ndim=2] _output = np.empty(z.shape, dtype=DTYPE)
    cdef DTYPE_t[:,:] output = _output

    cdef DTYPE_t[:,:,:] xs_mv = xs
    cdef DTYPE_t[:,:] z_mv = z

    cdef int sx = z.shape[0]
    cdef int sy = z.shape[1]
    cdef int x0, x1
    cdef DTYPE_t px0, px1
    cdef DTYPE_t sxmax = sx-1-eps
    cdef DTYPE_t symax = sy-1-eps
    cdef int i, j
    cdef DTYPE_t a, intp
    cdef int fill = fill_value == None
    cdef DTYPE_t ctype_fill_value
    cdef DTYPE_t xp1, xp2
    cdef DTYPE_t eps = 1e-10
    if fill_value:
        ctype_fill_value = fill_value
    with nogil:
        for x0 in range(sx):
            for x1 in range(sy):
                px0 = startx0 + xs_mv[x0, x1, 0] / dx0
                px1 = startx1 + xs_mv[x0, x1, 1] / dx1
                if fill:
                    if px0 < 0.0: px0 = 0.0
                    elif px0 > sxmax: px0 = sxmax

                    if px1 < 0.0: px1 = 0.0
                    elif px1 > symax: px1 = symax
                else:
                    if dabs(px0-(sx-1)) < 0.0001:
                        px0 = sx-1-eps 
                    if dabs(px1-(sy-1)) < 0.0001:
                        px1 = sy-1-eps
            
                if 0.0 <= px0 < sx-1 and 0.0 <= px1 < sy-1:
                    i = <int>px0
                    j = <int>px1
                    a = px0-i
                    xp1 = lerp(a, z_mv[i,j], z_mv[i+1,j])
                    xp2 = lerp(a, z_mv[i,j+1], z_mv[i+1,j+1])
                    a = px1-j
                    intp = lerp(a, xp1, xp2)
                else: 
                    intp = ctype_fill_value

                output[x0,x1] = intp
    return _output 
    

    
