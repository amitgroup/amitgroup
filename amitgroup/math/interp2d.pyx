
# cython: boundscheck=True
# cython: wraparound=False
# cython: embedsignature=True
# cython: cdivision=True
# cython: profile=True
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

def interp2d(np.ndarray[DTYPE_t, ndim=2] x, np.ndarray[DTYPE_t, ndim=2] y, z, dx=None, startx=None, fill_value=None): 
    """
    Calculates bilinear interpolated points of ``z`` at positions ``x`` and ``y``.
    
    The motivation of this function is that ``scipy.interpolate.interp2d(..., kind='linear')`` produces unwanted results.

    Parameters
    ----------
    x, y : ndarray
        Points at which to interpolate data. Array of shape ``(A, B)``, where ``A`` and ``B`` are the rows and columns.
    z : ndarray
        The original array that should be interpolated. Array of size ``(A, B)``.
    dx : ndarray or None
        The distance between points in ``z``. Array of size 2.
        If None, even spacing that range from 0.0 to 1.0 is assumed.
    startx : ndarray or None
        The ``(x, y)`` value corresponding to ``z[0,0]``. Array of size 2. 
    fill_value : float or None
        The value to return outside the area specified by ``z``. If None, the closest value inside the area is used.

    Returns
    -------
    output : ndarray
        Array of shape ``(A, B)`` with interpolated values at positions at ``x`` and ``y``.
    """
    
    assert(x.dtype == DTYPE)
    assert(y.dtype == DTYPE)
    assert(z.dtype == DTYPE)
    assert(x.shape == y.shape, "x and y must be the same shape")
    dx = dx if dx is not None else 1.0/(np.array(z.shape))
    startx = startx if startx is not None else np.zeros(2) 

    cdef:
        DTYPE_t startx0 = startx[0]
        DTYPE_t startx1 = startx[1]
        DTYPE_t dx0 = dx[0]
        DTYPE_t dx1 = dx[1]

        int sx0 = x.shape[0]
        int sx1 = x.shape[1]
        int sz0 = z.shape[0]
        int sz1 = z.shape[1]
        np.ndarray[DTYPE_t, ndim=2] _output = np.empty((sx0, sx1), dtype=DTYPE)
        DTYPE_t[:,:] output = _output

        DTYPE_t[:,:] x_mv = x
        DTYPE_t[:,:] y_mv = y
        DTYPE_t[:,:] z_mv = z

        int x0, x1
        DTYPE_t eps = 1e-10
        DTYPE_t pz0, pz1
        DTYPE_t sz0max = sz0-1-eps
        DTYPE_t sz1max = sz1-1-eps
        int i, j
        DTYPE_t a, intp
        int fill = fill_value == None
        DTYPE_t ctype_fill_value = 0.0
        DTYPE_t xp1, xp2

    if fill_value:
        ctype_fill_value = fill_value
    with nogil:
        for x0 in range(sx0):
            for x1 in range(sx1):
                pz0 = startx0 + x_mv[x0, x1] / dx0
                pz1 = startx1 + y_mv[x0, x1] / dx1
                if fill:
                    if pz0 < 0.0: pz0 = 0.0
                    elif pz0 > sz0max: pz0 = sz0max

                    if pz1 < 0.0: pz1 = 0.0
                    elif pz1 > sz1max: pz1 = sz1max
                else:
                    if dabs(pz0-(sz0-1)) < 0.0001:
                        pz0 = sz0-1-eps 
                    if dabs(pz1-(sz1-1)) < 0.0001:
                        pz1 = sz1-1-eps
            
                if 0.0 <= pz0 < sz0-1 and 0.0 <= pz1 < sz1-1:
                    i = <int>pz0
                    j = <int>pz1
                    a = pz0-i
                    xp1 = lerp(a, z_mv[i,j], z_mv[i+1,j])
                    xp2 = lerp(a, z_mv[i,j+1], z_mv[i+1,j+1])
                    a = pz1-j
                    intp = lerp(a, xp1, xp2)
                else: 
                    intp = ctype_fill_value

                output[x0,x1] = intp
    return _output 
    