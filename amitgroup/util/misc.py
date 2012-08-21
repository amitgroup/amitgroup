
import numpy as np

def zeropad(data, padwidth):
    data = np.asarray(data)
    shape = data.shape
    if isinstance(padwidth, int):
        padwidth = (padwidth,)*len(shape) 
        
    padded_shape = map(lambda ix: ix[1]+padwidth[ix[0]]*2, enumerate(shape))
    new_data = np.zeros(padded_shape)
    new_data[ [slice(w, -w) if w > 0 else slice(None) for w in padwidth] ] = data 
    return new_data
