
from __future__ import division
import numpy as np
import scipy.stats
import math

def visualize_hog(hog_features, cell_size, show=True):
    import matplotlib.pylab as plt

    # We'll construct an image and then show it
    img = np.zeros(tuple([hog_features.shape[i] * cell_size[i] for i in xrange(2)]))

    num_bins = hog_features.shape[2] 

    x, y = np.mgrid[-0.5:0.5:cell_size[0]*1j, -0.5:0.5:cell_size[1]*1j]
    circle = (x**2 + y**2) <= 0.28

    arrows = np.empty((num_bins,) + cell_size)
    
    # Generate the lines that will indicate directions
    for d in xrange(num_bins):
        # v is perpendicular to the gradient (thus visualizing an edge)
        v = np.array([math.cos(math.pi/2 + d*2*math.pi/num_bins), -math.sin(math.pi/2 + d*2*math.pi/num_bins)])
        # project our location onto this line and run that through a guassian (basically, drawing a nice line)
        arrows[d] = scipy.stats.norm.pdf(v[0] * x + v[1] * y, scale=0.05)

    arrows[:] *= circle
    
    for x in xrange(hog_features.shape[0]):
        for y in xrange(hog_features.shape[1]):
            for angle in xrange(num_bins):
                img[x*cell_size[0]:(x+1)*cell_size[0],y*cell_size[1]:(y+1)*cell_size[1]] += arrows[angle] * hog_features[x, y, angle]
    
    plt.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    
    if show:
        plt.show()
