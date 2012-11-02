
import numpy as np
import scipy.signal as sig

# TODO: Under development
def hog(data, cell_size, block_size):
    kernel = np.array([[-1, 0, 1]])
    grad1, grad2 = sig.convolve(data, kernel, mode='same'), sig.convolve(data, kernel.T, mode='same')
    
    grad1[:,0] = grad1[:,-1] = grad2[0] = grad2[-1] = 0

    angles = np.arctan2(grad2, grad1)

    # Build a kernel that will 
    x, y = np.mgrid[-0.5:0.5:cell_size[0]*1j, -0.5:0.5:cell_size[1]*1j]
    
    bikernel = np.fabs(x * y)
    
    #print bikernel
