
import numpy as np

def zeropad(data, padwidth):
    shape = data.shape
    padded_shape = map(lambda x: x+padwidth*2, shape)
    new_data = np.zeros(padded_shape)
    new_data[ [slice(padwidth, -padwidth)]*len(shape) ] = data 
    return new_data
