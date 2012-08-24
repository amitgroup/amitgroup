from __future__ import division
import timeit
import numpy as np
import amitgroup as ag
import pywt
from time import time
import amitgroup.util.wavelet


def run_test(shape, levels):
    setup = "x = np.arange(np.prod({0})).reshape({0})".format(shape)
    N = 10
    #u = pywt.wavedec2(x, 'db2', mode='per', level=5)

    max_level = int(np.log2(max(shape)))

    setup = """
import pywt
import amitgroup.util.wavelet
import numpy as np
import amitgroup as ag
x = np.arange(np.prod({0})).reshape({0})
wavedec2 = ag.util.wavelet.daubechies2d_forward_factory({0}, levels={1})
waverec2 = ag.util.wavelet.daubechies2d_inverse_factory({0}, levels={1})
u = pywt.wavedec2(x, 'db2', mode='per', level={1})
u2 = wavedec2(x)
    """.format(shape, max_level)

    wavedec2_pywt = timeit.Timer("u = pywt.wavedec2(x, 'db2', mode='per', level=5)", setup).timeit(N)/N
    wavedec2_amit = timeit.Timer("coefs = wavedec2(x)", setup).timeit(N)/N 

    waverec2_pywt = timeit.Timer("pywt.waverec2(u, 'db2', mode='per')", setup).timeit(N)/N
    waverec2_amit = timeit.Timer("A = waverec2(x)", setup).timeit(N)/N

    print "Shape: {0}, levels used: ({1})".format(shape, levels)
    print "pywt wavedec2d:", 1000*wavedec2_pywt, "ms"
    print "amit wavedec2d:", 1000*wavedec2_amit, "ms"
    print "pywt waverec2d:", 1000*waverec2_pywt, "ms"
    print


if __name__ == '__main__':
    #for shapes in [(16, 16), (32, 32), (64, 64), (256, 256), (1024, 1024)]:
    for shapes in [(32, 32)]:
        max_level = int(np.log2(max(shapes)))
        levels = max_level//2
        #for levels in xrange(max_level, max_level+1):
        run_test(shapes, levels)
