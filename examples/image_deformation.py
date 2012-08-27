
import amitgroup as ag
import numpy as np
from time import time

def main():
    F, I = ag.io.load_example('two-faces')

    t1 = time()
    imdef, info = ag.stats.image_deformation(F, I, penalty=300.0, rho=2.0, last_level=3, tol=0.00001, maxiter=100, \
                                          start_level=1, wavelet='db8', debug_plot=False)
    t2 = time()
    Fdef = imdef.deform(F)

    print "Time:", t2-t1
    print "Cost:", info['cost']

    PLOT = True 
    if PLOT:
        import matplotlib.pylab as plt
        ag.plot.deformation(F, I, imdef)

if __name__ == '__main__':
    #import cProfile as profile
    #profile.run('main()')
    main()
