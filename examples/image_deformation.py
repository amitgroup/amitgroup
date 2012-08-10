
import amitgroup as ag
import amitgroup.ml
import numpy as np
from time import time

def main():
    t1 = time()
    F, I = ag.io.load_example('two-faces')

    imdef, info = ag.ml.image_deformation(F, I, stepsize_scale_factor=1.0, penalty=0.1, rho=1.0, last_level=4, tol=0.001, \
                                          start_level=2, wavelet='db2')
    Fdef = imdef.deform(F)
    t2 = time()

    print "Time:", t2-t1

    PLOT = True
    if PLOT:
        import matplotlib.pylab as plt
        x, y = imdef.meshgrid()
        Ux, Uy = imdef.deform_map(x, y) 

        ag.plot.deformation(F, I, imdef)

        #Also print some info before showing
        print "Iterations (per level):", info['iterations_per_level'] 
        
        plt.show()

        # Print 
        plt.semilogy(info['costs'])
        plt.show()

if __name__ == '__main__':
    #import cProfile as profile
    #profile.run('main()')
    main()
