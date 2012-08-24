from __future__ import division
import numpy as np
import amitgroup as ag
import pywt
from time import time

x = np.arange(32*32).reshape((32, 32))

imdef = ag.util.DisplacementFieldWavelet((32, 32), 'db2', level_capacity=3)
db4g = np.array([0.48296291314469025, 0.83651630373746899, 0.22414386804185735, -0.12940952255092145])
db4h = db4g[::-1].copy()
db4h[1::2] *= -1
N = len(db4g)

def qdot(X, A):
    return np.dot(X, np.dot(A, X.T))

def top_left_quad(A):
    N = len(A)//2
    return A[:N,:N]
def top_right_quad(A):
    N = len(A)//2
    return A[:N,N:]
def bottom_right_quad(A):
    N = len(A)//2
    return A[N:,N:]
def bottom_left_quad(A):
    N = len(A)//2
    return A[N:,:N]

def populate(W, filtr, yoffset):
    N = len(filtr)
    for i in range(W.shape[1]//2):
        for j in range(N):
            W[yoffset+i, (-(N-2)//2+2*i+j)%W.shape[1]] += filtr[j]


    if 0:
        fr, to = -1+2*i, -1+len(filtr)+2*i
        fr0, to0 = max(0, fr), min(W.shape[0], to)
        if fr != fr0:
            W[yoffset+i,fr:] = filtr[:-fr]
        if to != to0:
            W[yoffset+i,:to-W.shape[0]] = filtr[-(to-W.shape[0]):]
        W[yoffset+i,fr0:to0] = filtr[fr0-fr:fr0-fr+(to0-fr0)]

W = np.zeros((32, 32))
populate(W, db4g, 0)
populate(W, db4h, 16)

W2 = np.zeros((16, 16))
populate(W2, db4g, 0)
populate(W2, db4h, 8)

W3 = np.zeros((8, 8))
populate(W3, db4g, 0)
populate(W3, db4h, 4)

W4 = np.zeros((4, 4))
populate(W4, db4g, 0)
populate(W4, db4h, 2)

W5 = np.zeros((2, 2))
populate(W5, db4g, 0)
populate(W5, db4h, 1)

G1 = np.zeros((16, 32))
G2 = np.zeros((8, 16))

populate(G1, db4g, 0)
populate(G2, db4g, 0)
WG = np.dot(W3, np.dot(G2, G1))

def wavedec2_32(A):
    Ab = qdot(WG, A) 
    Abc = Ab.copy()
    B = top_left_quad(Ab)
    Bb = qdot(W4, B)
    C = top_left_quad(Bb)
    Cb = qdot(W5, C)
    D = top_left_quad(Cb)

    Ab[:4,:4] = Bb
    Ab[:2,:2] = Cb
    Ab[0,0] = D[0,0]
    return Ab

    #np.testing.assert_array_almost_equal(Ab, Abc)

    #imdef = ag.util.DisplacementFieldWavelet((32, 32), 'db2')

    #res = np.empty(64)
    #res[:4] = Cc.T.flatten()
    #res[4:8] = bottom_left_quad(Bb).flatten() 
    #res[8:12] = top_right_quad(Bb).flatten() 
    #res[12:16] = bottom_right_quad(Bb).flatten() 
    #res[16:32] = bottom_left_quad(Ab).flatten()
    #res[32:48] = top_right_quad(Ab).flatten()
    #res[48:64] = bottom_right_quad(Ab).flatten()
    #return res

N = 1

t1 = time()
for i in range(N):
    u = pywt.wavedec2(x, 'db2', mode='per', level=5)
    coefs = ag.util.DisplacementFieldWavelet.pywt2array(u, imdef.levelshape, 3, 3)
t2 = time()

t1b = time()
for i in range(N):
    pywt.waverec2(u, 'db2', mode='per')
t2b = time()

t1c = time()
for i in range(N):
    coefs2 = wavedec2_32(x)
t2c = time()

import amitgroup.util.wavelet
wavedec2 = ag.util.wavelet.daubechies2d_forward_factory((32, 32), levels=3)

t1d = time()
for i in range(N):
    coefs2 = wavedec2(x)
t2d = time()

#print u
print 'coefs'
print coefs
print 'coefs2'
print ag.util.wavelet.new2old(coefs2)

np.testing.assert_array_almost_equal(coefs, ag.util.wavelet.new2old(coefs2))

np.testing.assert_array_almost_equal(ag.util.wavelet.old2new(ag.util.wavelet.new2old(coefs2)), coefs2)

print (t2-t1)/(t2c-t1c)
print "Deconstruction time:", 1000*(t2-t1)/N, "ms"
print "Deconstruction mine:", 1000*(t2c-t1c)/N, "ms"
print "Deconstruction new :", 1000*(t2d-t1d)/N, "ms"
print "Reconstruction time:", 1000*(t2b-t1b)/N, " ms"
