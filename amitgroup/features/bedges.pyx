#!python
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

cdef inline void checkedge(DTYPE_t[:,:,:] images, np.uint8_t[:,:,:,:] ret, int ii, int vi, int z0, int z1, int v0, int v1, int w0, int w1, int k) nogil:
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
    #if  d > dabs(images[ii, z0+w0, z1+w1] - Iz) and \
    #    d > dabs(images[ii, y0+w0, y1+w1] - Iy) and \
    #    d > dabs(images[ii, z0-w0, z1-w1] - Iz) and \
    #    d > dabs(images[ii, y0-w0, y1-w1] - Iy) and \
    #    d > dabs(images[ii, z0-v0, z1-v1] - Iz) and \
    #    d > dabs(images[ii, y0+v0, y1+v1] - Iy):
    if num_edges >= k:
        ret[ii, z0, z1, vi + 4*<int>(Iy > Iz)] = 1 

def _array_bedges(np.ndarray[DTYPE_t, ndim=3] images, k):
    assert(images.dtype == DTYPE)
    cdef int N = images.shape[0]
    cdef int rows = images.shape[1]
    cdef int cols = images.shape[2] 
    cdef np.ndarray[np.uint8_t, ndim=4] ret = np.zeros((N, rows, cols, 8), dtype=np.uint8)
    cdef np.uint8_t[:,:,:,:] ret_mv = ret
    cdef DTYPE_t[:,:,:] images_mv = images
    cdef Py_ssize_t i
    cdef int z0
    cdef int z1
    cdef int int_k = <int>k
    #for i in prange(N, nogil=True):
    with nogil:
        for i in xrange(N):
            for z0 in range(2, rows-2):
                for z1 in range(2, cols-2):
                    checkedge(images_mv, ret_mv, i, 0, z0, z1, 1, 0, 0, -1, int_k)
                    checkedge(images_mv, ret_mv, i, 1, z0, z1, 1, 1, 1, -1, int_k)
                    checkedge(images_mv, ret_mv, i, 2, z0, z1, 0, 1, 1, 0, int_k)
                    checkedge(images_mv, ret_mv, i, 3, z0, z1, -1, 1, 1, 1, int_k)

    return ret

def bedges(images, k=6, features_as_axis0=False):
    """
    Extracts binary edge features for each pixel according to [1].

    Parameters
    ----------
    images : ndarray
        Input an image of shape ``(rows, cols)`` or a list of images as an array of shape ``(N, rows, cols)``, where ``N`` is the number of images, and ``rows`` and ``cols`` the size of each image.
    k : int
        Number of features out of 6 for something to be an edge. Default is 6. Lower it if you want more edge features extracted.

    Returns
    -------
    edges : ndarray
        An array of shape ``(rows, cols, 8)`` if entered as a single image, or ``(N, rows, cols, 8)`` of multiple. Each pixel in the original image becomes a binary vector of size 8, one bit for each cardinal and diagonal direction. 

    References
    ----------
    [1] Y. Amit : 2D Object Detection and Recognition: Models, Algorithms and Networks. Chapter 5.4.
    """
    assert 0 < k <= 6, "k must be less than or equal to 6"
    single = len(images.shape) == 2
    if single:
        return _array_bedges(np.array([images]), k)[0]
    else:
        return _array_bedges(images, k) 

      
