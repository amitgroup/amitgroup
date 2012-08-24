# Aims to provide functions for fast periodic daubechies transforms (forward and inverse) in 2D

import numpy as np

def _populate(W, filtr, yoffset):
    N = len(filtr)
    for i in range(W.shape[1]//2):
        for j in range(N):
            W[yoffset+i, (-(N-2)//2+2*i+j)%W.shape[1]] += filtr[j]

def _create_W(shape, level, filter_low, filter_high):
    d = 2**(level-1)
    W = np.zeros((shape[0]//d, shape[1]//d))
    _populate(W, filter_low, 0)
    _populate(W, filter_high, shape[0]//(2*d))
    return W

def _create_single(shape, level, filtr):
    d = 2**(level-1)
    GH = np.zeros((shape[0]//(2*d), shape[1]//d))
    _populate(GH, filtr, 0)
    return GH

def _qdot(X, A):
    return np.dot(X, np.dot(A, X.T))

def _top_left_quad(A):
    N = len(A)//2
    return A[:N,:N]
def _top_right_quad(A):
    N = len(A)//2
    return A[:N,N:]
def _bottom_right_quad(A):
    N = len(A)//2
    return A[N:,N:]
def _bottom_left_quad(A):
    N = len(A)//2
    return A[N:,:N]

def daubechies2d_forward_factory(shape, levels=3):
    db4g = np.array([0.48296291314469025, 0.83651630373746899, 0.22414386804185735, -0.12940952255092145])
    db4h = db4g[::-1].copy()
    db4h[1::2] *= -1

    filter_low = db4g
    filter_high = db4h

    assert len(shape) == 2
    assert shape[0] == shape[1], "Shape must be square (at least for now)"

    levels_list = range(1, int(np.log2(shape[0]))+1)

    # Setup matrices
    Ws = [_create_W(shape, level, filter_low, filter_high) for level in levels_list]
    Gs = [_create_single(shape, level, filter_low) for level in levels_list]
    # Combine all the matrices for the steps where we throw away the coefficients.
    Wg = Gs[0]
    for l in xrange(1, levels-1):
        Wg = np.dot(Gs[l], Wg)
    # Also include the first Wavelet transform.
    Wg = np.dot(Ws[levels-1], Wg)

    def daubechies2d_forward(A, level):
        assert A.shape == (32, 32)
        Ab = _qdot(Wg, A) 
        Abc = Ab.copy()
        B = _top_left_quad(Ab)
        Bb = _qdot(Ws[3], B)
        C = _top_left_quad(Bb)
        Cb = _qdot(Ws[4], C)
        #D = _top_left_quad(Cb)

        R = np.zeros(Ab.shape)
        
        if level >= 3:
            R[:] = Ab
        if level >= 2:
            R[:4,:4] = Bb
        if level >= 1:
            R[:2,:2] = Cb
        if level >= 0:
            R[0,0] = Cb[0,0]
             
        return R
    return daubechies2d_forward
        
def daubechies2d_inverse_factory(shape, levels=3):
    db4g = np.array([0.48296291314469025, 0.83651630373746899, 0.22414386804185735, -0.12940952255092145])
    db4h = db4g[::-1].copy()
    db4h[1::2] *= -1

    filter_low = db4g
    filter_high = db4h

    assert len(shape) == 2
    assert shape[0] == shape[1], "Shape must be square (at least for now)"

    levels_list = range(1, int(np.log2(shape[0]))+1)

    # Setup matrices
    Ws = [_create_W(shape, level, filter_low, filter_high) for level in levels_list]
    Gs = [_create_single(shape, level, filter_low) for level in levels_list]
    # Combine all the matrices for the steps where we throw away the coefficients.
    Wg = Gs[0]
    for l in xrange(1, levels-1):
        Wg = np.dot(Gs[l], Wg)
    # Also include the first Wavelet transform.
    Wg = np.dot(Ws[levels-1], Wg)
    
    def daubechies2d_inverse(R, level):
        #print R.shape
        R = R.copy().reshape(8, 8)
        #assert R.shape == (8, 8)

        R[:2,:2] = _qdot(Ws[4].T, R[:2,:2])
        R[:4,:4] = _qdot(Ws[3].T, R[:4,:4])
        #R[:8,:8] = qdot(Ws[2].T
        R = _qdot(Wg.T, R)
        #C = qdot(W
        return R

    return daubechies2d_inverse


# TODO: Make this general to any size (starting with squares)
def new2old(news):
    assert news.shape == (8, 8), "Not {0}".format(news.shape)
    olds = np.zeros(64)
    olds[0] = news[0,0]
    olds[1] = news[1,0]
    olds[2] = news[0,1]
    olds[3] = news[1,1]
    olds[4:8] = news[2:4,0:2].flatten()
    olds[8:12] = news[0:2,2:4].flatten()
    olds[12:16] = news[2:4,2:4].flatten()
    olds[16:32] = news[4:8,0:4].flatten()
    olds[32:48] = news[0:4,4:8].flatten()
    olds[48:64] = news[4:8,4:8].flatten()
    return olds 

def old2new(olds):
    N = int(np.sqrt(len(olds)))
    A = np.arange(N*N, dtype=int).reshape(N, N)
    indices = new2old(A).astype(int)
    new_indices = np.empty(indices.shape, dtype=int)
    for i, index in enumerate(indices):
        new_indices[index] = i
    news = olds[new_indices].reshape(8, 8).copy()
    return news 

