
import amitgroup as ag
import matplotlib.pylab as plt
import numpy as np

data = plt.imread('circle.png')
data = data[...,:3].mean(axis=-1)


feat = ag.features.hog(data, (6, 6), (2, 2), num_bins=12)

plt.subplot(121)
plt.imshow(data)
plt.subplot(122)
ag.plot.visualize_hog(feat, (21, 21), show=False)
plt.show()
