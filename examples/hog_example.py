
import amitgroup as ag
import matplotlib.pylab as plt
import numpy as np

data = plt.imread('circle.png')
data = data[...,:3].mean(axis=-1)

feat = ag.features.hog(data, (12, 12), (2, 2), num_bins=36)

ag.plot.visualize_hog(feat, (11, 11))
