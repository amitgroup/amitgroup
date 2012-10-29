#!python
# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature=True
import cython
import numpy as np
cimport numpy as np
#from cython.parallel import prange
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef inline DTYPE_t dabs(DTYPE_t x) nogil: 
    return x if x >= 0 else -x 

cdef inline void _checkedge(DTYPE_t[:,:,:] images, np.uint8_t[:,:,:,:] ret, int ii, int vi, int z0, int z1, int v0, int v1, int w0, int w1, int k, double minimum_contrast, int displace) nogil:
    cdef int y0 = z0 + v0
    cdef int y1 = z1 + v1
    cdef DTYPE_t m
    cdef DTYPE_t Iy = images[ii, y0, y1] 
    cdef DTYPE_t Iz = images[ii, z0, z1] 
    
    cdef DTYPE_t d = dabs(Iy - Iz)
    cdef int num_edges = <int>(d > dabs(images[ii, z0+w0, z1+w1] - Iz)) + \
                         <int>(d > dabs(images[ii, y0+w0, y1+w1] - Iy)) + \
                         <int>(d > dabs(images[ii, z0-w0, z1-w1] - Iz)) + \
                         <int>(d > dabs(images[ii, y0-w0, y1-w1] - Iy)) + \
                         <int>(d > dabs(images[ii, z0-v0, z1-v1] - Iz)) + \
                         <int>(d > dabs(images[ii, y0+v0, y1+v1] - Iy))

    if num_edges >= k and d > minimum_contrast: 
        ret[ii, vi + displace*<int>(Iy > Iz), z0, z1] = 1 

def array_bedges(np.ndarray[DTYPE_t, ndim=3] images, k, minimum_contrast, contrast_insensitive):
    assert(images.dtype == DTYPE)
    cdef int N = images.shape[0]
    cdef int rows = images.shape[1]
    cdef int cols = images.shape[2] 
    cdef DTYPE_t[:,:,:] images_mv = images
    cdef Py_ssize_t i
    cdef int z0
    cdef int z1
    cdef int int_k = <int>k
    cdef double double_minimum_contrast = <double>minimum_contrast
    cdef int displace = 0
    cdef int binary_features = 8

    if contrast_insensitive:
        displace = 0
        binary_features = 4
    else:
        displace = 4 
        binary_features = 8

    cdef np.ndarray[np.uint8_t, ndim=4] ret = np.zeros((N, binary_features, rows, cols), dtype=np.uint8)
    cdef np.uint8_t[:,:,:,:] ret_mv = ret
    
    #for i in prange(N, nogil=True):
    with nogil:
        for i in xrange(N):
            for z0 in range(2, rows-2):
                for z1 in range(2, cols-2):
                    _checkedge(images_mv, ret_mv, i, 0, z0, z1, 1, 0, 0, -1, int_k, double_minimum_contrast, displace)
                    _checkedge(images_mv, ret_mv, i, 1, z0, z1, 1, 1, 1, -1, int_k, double_minimum_contrast, displace)
                    _checkedge(images_mv, ret_mv, i, 2, z0, z1, 0, 1, 1, 0, int_k, double_minimum_contrast, displace)
                    _checkedge(images_mv, ret_mv, i, 3, z0, z1, -1, 1, 1, 1, int_k, double_minimum_contrast, displace)

    return ret
