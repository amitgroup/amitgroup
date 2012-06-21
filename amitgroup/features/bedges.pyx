import cython
import numpy as np
cimport numpy as np
from cython.parallel import prange
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef inline DTYPE_t dabs(DTYPE_t x) nogil: 
    return x if x >= 0 else -x 

@cython.boundscheck(False)
cdef inline void checkedge(DTYPE_t[:,:,:] images, np.uint8_t[:,:,:,:] ret, int ii, int vi, int z0, int z1, int v0, int v1, int w0, int w1) nogil:
    cdef int y0 = z0 + v0
    cdef int y1 = z1 + v1
    cdef DTYPE_t m
    cdef DTYPE_t Iy = images[ii, y0, y1] 
    cdef DTYPE_t Iz = images[ii, z0, z1] 
    
    cdef DTYPE_t d = dabs(Iy - Iz)
    if  d > dabs(images[ii, z0+w0, z1+w1] - Iz) and \
        d > dabs(images[ii, y0+w0, y1+w1] - Iy) and \
        d > dabs(images[ii, z0-w0, z1-w1] - Iz) and \
        d > dabs(images[ii, y0-w0, y1-w1] - Iy) and \
        d > dabs(images[ii, z0-v0, z1-v1] - Iz) and \
        d > dabs(images[ii, y0+v0, y1+v1] - Iy):
        ret[ii, z0, z1, vi + 4*<int>(Iy > Iz)] = 1 

@cython.boundscheck(False)
def bedges(np.ndarray[DTYPE_t, ndim=3] _images):
    """
    Extracts binary edge features for each pixel according to [1].

    Parameters
    ----------
    images : ndarray
        Input all images as an ndarray of shape ``(N, rows, cols)``, where ``N`` is the number of images, and ``rows`` and ``cols`` the size of each image.

    Returns
    -------
    edges : ndarray
        An array of shape ``(N, rows, cols, 8)``, where each pixel in the original image becomes a binary vector of size 8, one bit for each cardinal and diagonal direction. 

    References
    ----------
    [1] Y. Amit : 2D Object Detection and Recognition: Models, Algorithms and Networks. Chapter 5.4.
    """
    assert(_images.dtype == DTYPE)
    cdef int N = _images.shape[0]
    cdef int rows = _images.shape[1]
    cdef int cols = _images.shape[2] 
    cdef np.ndarray[np.uint8_t, ndim=4] _ret = np.zeros((N, rows, cols, 8), dtype=np.uint8)
    cdef np.uint8_t[:,:,:,:] ret = _ret
    cdef DTYPE_t[:,:,:] images = _images
    cdef Py_ssize_t i
    cdef int z0
    cdef int z1
    for i in prange(N, nogil=True):
    #for i in xrange(N):
        for z0 in range(2, rows-2):
            for z1 in range(2, cols-2):
                checkedge(images, ret, i, 0, z0, z1, 1, 0, 0, -1)
                checkedge(images, ret, i, 1, z0, z1, 1, 1, 1, -1)
                checkedge(images, ret, i, 2, z0, z1, 0, 1, 1, 0)
                checkedge(images, ret, i, 3, z0, z1, -1, 1, 1, 1)

    return _ret
      

