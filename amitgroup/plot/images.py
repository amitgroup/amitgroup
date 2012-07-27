
import numpy as np
import matplotlib.pylab as plt
import math

def images(data, zero_to_one=True):
    """
    Display images that range from 0 to 1 in a grid. 

    Parameters
    ----------
    data : ndarray
        An array of images or a single image of shape. The values should range between 0 and 1 (at least, that is how they will be colorized).
    """

    settings = {
        'interpolation': 'nearest',
        'cmap': plt.cm.gray,
    }

    if zero_to_one:
        settings['vmin'] = 0.0
        settings['vmax'] = 1.0

    if isinstance(data, np.ndarray) and len(data.shape) == 2:
        fig = plt.figure()
        plt.subplot(111).set_axis_off()
        plt.imshow(data, **settings)
    else:
        # TODO: Better find out pleasing aspect ratios
        if len(data) <= 3:
            sh = (1, len(data))
        if len(data) == 6:
            sh = (2, 3)
        if len(data) == 12:
            sh = (3, 4)
        else:
            perside = math.ceil(math.sqrt(len(data)))
            sh = (perside,)*2
        fig = plt.figure()
        for i, im in enumerate(data): 
            plt.subplot(sh[0], sh[1], 1+i).set_axis_off()
            plt.imshow(im, **settings)
             
        #sh = [(), (1,), (2,1), (2,2), 


