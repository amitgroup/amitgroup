
import amitgroup as ag
import matplotlib.pylab as plt
import numpy as np

origdata = plt.imread('circle.png')
data = origdata[...,:3].mean(axis=-1)


feat = ag.features.hog(data, (6, 6), (3, 3), num_bins=12)

plt.subplot(121)
plt.imshow(origdata)
plt.subplot(122)
ag.plot.visualize_hog(feat, (5, 5), show=False)
plt.show()
