
import numpy as np
import matplotlib.pylab as plt
import math

def images(data, zero_to_one=True):
    """
    Display images that range from 0 to 1 in a grid. 

    Parameters
    ----------
    data : ndarray
        An array of images of shape ``(N, rows, cols)``, or a single image of shape ``(rows, cols)``. The values should range between 0 and 1 (at least, that is how they will be colorized).
    """
    assert 2 <= len(data.shape) <= 3

    settings = {
        'interpolation': 'nearest',
        'cmap': plt.cm.gray,
    }

    if zero_to_one:
        settings['vmin'] = 0.0
        settings['vmax'] = 1.0

    if len(data.shape) == 2:
        fig = plt.figure()
        plt.subplot(111).set_axis_off()
        plt.imshow(data, **settings)
    else:
        perside = math.ceil(math.sqrt(len(data)))
        sh = (perside,)*2
        fig = plt.figure()
        for i, im in enumerate(data): 
            plt.subplot(sh[0], sh[1], 1+i).set_axis_off()
            plt.imshow(im, **settings)
             
        #sh = [(), (1,), (2,1), (2,2), 


