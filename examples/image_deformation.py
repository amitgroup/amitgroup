
import amitgroup as ag
import amitgroup.ml
import numpy as np
import matplotlib.pylab as plt
from time import time

t1 = time()
F, I = ag.io.load_example('faces2')
imgdef, info = ag.ml.imagedef(F, I, stepsize=0.001, coef=0.1, rho=1.3, A=4, calc_costs=True)
Fdef = imgdef.deform(F)
t2 = time()

print t2-t1

if 1:
    x, y = imgdef.get_x(F.shape)
    Ux, Uy = imgdef.deform_map(x, y) 

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

    plt.show()

    # Print 
    plt.semilogy((-info['loglikelihoods']-info['logpriors']))
    plt.show()
