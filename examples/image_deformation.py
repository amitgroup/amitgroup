
import amitgroup as ag
import amitgroup.ml
import numpy as np
from time import time

def main():
    t1 = time()
    F, I = ag.io.load_example('faces2')

    imdef, info = ag.ml.imagedef(F, I, stepsize=None, penalty=0.1, rho=1.0, A=4, tol=0.001, \
                                 start_level=2, calc_costs=True, wavelet='db2')
    Fdef = imdef.deform(F)
    t2 = time()

    print t2-t1

    if 1:
        import matplotlib.pylab as plt
        x, y = imdef.get_x(F.shape)
        Ux, Uy = imdef.deform_map(x, y) 

        d = dict(interpolation='nearest', cmap=plt.cm.gray)
        plt.figure(figsize=(7,7))
        plt.subplot(221)
        plt.title("Prototype")
        plt.imshow(F, **d)
        plt.subplot(222)
        plt.title("Data image")
        plt.imshow(I, **d) 
        plt.subplot(223)
        plt.title("Deformed")
        plt.imshow(Fdef, **d)
        plt.subplot(224)
        plt.title("Deformation map")
        plt.quiver(y, -x, Uy, -Ux)

        #Also print some info before showing
        print info['iterations_per_level'] 
        
        #print imdef.u[0,0,0,0,0], 0.35
        #print imdef.u[1,0,0,0,0], -0.1
        #print imdef.u[0,1,0,0,0], 0.15
        #print imdef.u[0,1,1,0,0], -0.2
        #print imdef.u[1,1,2,0,0], 0.3

        plt.show()

        # Print 
        plt.semilogy(info['costs'])
        plt.show()

if __name__ == '__main__':
    #import cProfile as profile
    #profile.run('main()')
    main()
