
import numpy as np
import scipy.signal as sig
from itertools import product
import math

# TODO: Under development
def hog(data, cell_size, block_size, num_bins=9):
    gradkernel = np.array([[-1, 0, 1]])

    grad1, grad2 = sig.convolve(data, gradkernel, mode='same'), sig.convolve(data, gradkernel.T, mode='same')
    
    grad1[:,0] = grad1[:,-1] = grad2[0] = grad2[-1] = 0

    angles = np.arctan2(grad2, grad1)
    amplitudes = np.sqrt(grad1**2 + grad2**2)

    # Build a kernel that will 
    cx, cy = np.mgrid[-0.5:0.5:cell_size[0]*1j, -0.5:0.5:cell_size[1]*1j]
    
    kernels = []
    # Iterates all combinations of (-) and (+) to generate the bilinearity kernels 
    for op1, op2 in product([np.subtract, np.add], repeat=2):
        kernels.append(np.fabs(op1(0.5, cx) * op2(0.5, cy)))
    kernels = np.asarray(kernels)

    cells_shape = tuple([data.shape[i]//cell_size[i] for i in xrange(2)])
    votes = np.zeros(cells_shape + (num_bins,))

    # I'm sure we can optimize this a lot. This is just to get it going.
    for x in xrange(cells_shape[0]-1):
        for y in xrange(cells_shape[1]-1):
            for i in xrange(cell_size[0]):
                for j in xrange(cell_size[1]):
                    ix, iy = cell_size[0]//2 + x*cell_size[0] + i, cell_size[1]//2 + y*cell_size[1] + j
                    v = amplitudes[ix, iy]
                    angle = angles[ix, iy]
                    angle_bin = int(round(angle*num_bins/(2*math.pi))) % num_bins
                    votes[x,y,angle_bin] += kernels[0,i,j] * v
                    votes[x+1,y,angle_bin] += kernels[1,i,j] * v 
                    votes[x,y+1,angle_bin] += kernels[2,i,j] * v
                    votes[x+1,y+1,angle_bin] += kernels[3,i,j] * v
            
    return votes
