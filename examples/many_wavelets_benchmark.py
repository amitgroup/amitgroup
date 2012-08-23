from __future__ import division
import numpy as np
import pywt
from time import time

x = np.arange(32*32).reshape((32, 32))

t1 = time()
for i in range(1000):
    u = pywt.wavedec2(x, 'db4', mode='per', level=5)
t2 = time()

t1b = time()
for i in range(1000):
    pywt.waverec2(u, 'db4', mode='per')
t2b = time()

print u

print "Deconstruction time:", (t2-t1), " ms"
print "Reconstruction time:", (t2b-t1b), " ms"
