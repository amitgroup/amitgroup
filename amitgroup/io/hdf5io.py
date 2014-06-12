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
        new_group = handler.createGroup(group, name, "dict:{}".format(len(level)))
        for k, v in level.items():
            _save_level(handler, new_group, v, name=k) 
    elif isinstance(level, list):
        # Lists can contain other dictionaries and numpy arrays, so we don't want to
        # serialize them. Instead, we will store each entry as i0, i1, etc.
        new_group = handler.createGroup(group, name, "list:{}".format(len(level)))

        for i, entry in enumerate(level):
            level_name = 'i{}'.format(i)
            _save_level(handler, new_group, entry, name=level_name)

    elif isinstance(level, np.ndarray):
        atom = tables.Atom.from_dtype(level.dtype)
        node = handler.createCArray(group, name, atom=atom, shape=level.shape, chunkshape=level.shape, filters=COMPRESSION) 
        node[:] = level
    else:
        setattr(group._v_attrs, name, level)
        

def _load_level(level):
    if isinstance(level, tables.Group):
        dct = {} 
        # Load sub-groups
        for grp in level: 
            dct[grp._v_name] = _load_level(grp)

        # Load attributes
        for name in level._v_attrs._f_list():
            dct[name] = level._v_attrs[name]

        if level._v_title.startswith('list:'):
            N = int(level._v_title[len('list:'):])
            lst = []
            for i in range(N):
                lst.append(dct['i{}'.format(i)])
            return lst
        else:
            return dct
    elif isinstance(level, tables.Array):
        if level.shape == (1,):
            return level[0]
        else: 
            return level[:]

def save(path, data):
    """
    Save any Python structure to an HDF5 file. It is particularly suited for
    Numpy arrays. This function works similar to ``numpy.save``, except if you
    save a Python object at the top level, you do not need to issue
    ``data.flat[1]`` to retrieve it from inside a Numpy array of type
    ``object``.

    Four types objects get saved natively in HDF5, the rest get serialized. For
    most needs, you should be able to stick to the four, which are:

    * Dictionaries
    * Lists
    * Basic data types (including strings)
    * Numpy arrays

    A recommendation is to always convert your data to using only these four
    ingredients. That way your data will always be retrievable by any HDF5 
    reader.

    Parameters
    ---------- 
    path : file-like object or string
        File or filename to which the data is saved.
    data : anything
        Data to be saved. This can be anything from a Numpy array, a string, an
        object, or a dictionary containing all of them including more
        dictionaries.

    See also
    --------
    load 

    """
    if not isinstance(path, str):
        path = path.name

    h5file = tables.openFile(path, mode='w', title='Test file')
    #group = h5file.createGroup('/', 'detector', 'Detector information')
    group = h5file.root
    _save_level(h5file, group, data, name='top')
    h5file.close()

def load(path):
    """
    Load an HDF5 saved with `save`.

    Parameters
    ---------- 
    path : file-like object or string
        File or filename from which to load the data. 

    Returns 
    --------
    data : anything
        Hopefully an identical reconstruction of the data that was saved.

    See also
    --------
    save

    """
    if not isinstance(path, str):
        path = path.name

    h5file = tables.openFile(path, mode='r')
    data = _load_level(h5file.root)['top']
    h5file.close()
    return data 
