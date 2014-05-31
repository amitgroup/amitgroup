import numpy as np
import tables

try:
    COMPRESSION = tables.Filters(complevel=9, complib='blosc', shuffle=True)
except Exception: #type?
    warnings.warn("Missing BLOSC; no compression will used.")
    COMPRESSION = tables.Filters()

def _save_level(handler, group, level, name=None):
    if isinstance(level, dict):
        # First create a new group
        new_group = handler.createGroup(group, name)
        for k, v in level.items():
            _save_level(handler, new_group, v, name=k) 
    elif isinstance(level, np.ndarray):
        atom = tables.Atom.from_dtype(level.dtype)
        node = handler.createCArray(group, name, atom=atom, shape=level.shape, chunkshape=level.shape, filters=COMPRESSION) 
        node[:] = level
    #elif isinstance(level, int):
    #    atom = tables.Atom.from_dtype(np.dtype(np.int64))
    #    node = handler.createCArray(group, name, atom=atom, shape=(1,), chunkshape=(1,)) 
    #    node[0] = level
    #elif isinstance(level, float):
    #    atom = tables.Atom.from_dtype(np.dtype(np.float64))
    #    node = handler.createCArray(group, name, atom=atom, shape=(1,), chunkshape=(1,)) 
    #    node[0] = level
    else:
        atom = tables.Atom.from_dtype(np.dtype(type(level)))
        node = handler.createCArray(group, name, atom=atom, shape=(1,), chunkshape=(1,)) 
        node[0] = level
        

def save(filename, dct):
    h5file = tables.openFile(filename, mode='w', title='Test file')
    #group = h5file.createGroup('/', 'detector', 'Detector information')
    group = h5file.root
    _save_level(h5file, group, dct, name='top')
    h5file.close()

def _load_level(level):
    if isinstance(level, tables.Group):
        dct = {} 
        for grp in level: 
            dct[grp._v_name] = _load_level(grp)
        return dct
    elif isinstance(level, tables.Array):
        if level.shape == (1,):
            return level[0]
        else: 
            return level[:]

def load(filename):
    h5file = tables.openFile(filename, mode='r')
    dct = _load_level(h5file.root)['top']
    h5file.close()
    return dct 
