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


def daubechies2d_forward_factory(shape, end_level=4):
    db4g = np.array([0.48296291314469025, 0.83651630373746899, 0.22414386804185735, -0.12940952255092145])
    db4h = db4g[::-1].copy()
    db4h[1::2] *= -1

    filter_low = db4g
    filter_high = db4h

    assert len(shape) == 2
    assert shape[0] == shape[1], "Shape must be square (at least for now)"

    levels = range(1, int(np.log2(shape[0]))+1)

    # Setup matrices
    Ws = [_create_W(shape, level, filter_low, filter_high) for level in levels]

    Gs = [_create_single(shape, level, filter_low) for level in levels]

    Wg = np.dot(Ws[2], np.dot(Gs[1], Gs[0]))

    def daubechies2d_forward(A):
        Ab = _qdot(Wg, A) 
        Abc = Ab.copy()
        B = _top_left_quad(Ab)
        Bb = _qdot(Ws[3], B)
        C = _top_left_quad(Bb)
        Cb = _qdot(Ws[4], C)
        D = _top_left_quad(Cb)

        Ab[:4,:4] = Bb
        Ab[:2,:2] = Cb
        Ab[0,0] = D[0,0]
        return Ab
        
    return daubechies2d_forward

def daubechies2d_inverse_factory():
    pass
