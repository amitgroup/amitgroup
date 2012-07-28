
import amitgroup as ag
import amitgroup.ml
import numpy as np
import matplotlib.pylab as plt

F, I = ag.io.load_example('faces2')
imgdef, info = ag.ml.imagedef(F, I, coef=1e-4, rho=2.0, A=4, calc_costs=True)
Fdef = imgdef.deform(F)

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
x, y = imgdef.get_x(F.shape)
Ux, Uy = imgdef.deform_map(x, y) 
plt.quiver(y, -x, Uy, -Ux)

#Also print some info before showing
print info['iterations_per_level'] 

plt.show()

# Print 
plt.semilogy(-info['loglikelihoods']-info['logpriors'])
plt.show()
