
import numpy as np
import scipy.signal as sig
import scipy.stats
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

    # Now, sum these up over blocks and normalize
    blocks_shape = tuple([cells_shape[i]-block_size[i]+1 for i in xrange(2)])

    features = np.zeros(blocks_shape + (num_bins,))

    eps2 = 0.01

    # The minus one is important for centering the gaussian
    w, h = (blocks_shape[0]-1)/2, (blocks_shape[1]-1)/2
    for x in xrange(blocks_shape[0]):
        for y in xrange(blocks_shape[1]):
            v = np.zeros(votes.shape[-1])
            for i in xrange(block_size[0]):
                for j in xrange(block_size[1]):
                    v += votes[x+i,y+j] * scipy.stats.norm.pdf(np.sqrt((x-w)**2 + (y-h)**2), scale=0.5 * block_size[0])

            # Normalize and put into features 
            features[x,y] = v / np.sqrt(np.dot(v, v) + eps2)
            
    return features 
