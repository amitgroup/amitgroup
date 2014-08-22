from __future__ import division, print_function, absolute_import

import amitgroup as ag
import numpy as np
import tables
import warnings
import sys

from amitgroup import six

# Types that should be saved as pytables attribute
ATTR_TYPES = (int, float, bool, six.string_types,
              np.int8, np.int16, np.int32, np.int64,
              np.uint8, np.uint16, np.uint32, np.uint64,
              np.float16, np.float32, np.float64,
              np.bool_, np.complex64, np.complex128)

try:
    COMPRESSION = tables.Filters(complevel=9, complib='blosc', shuffle=True)
except Exception:
    warnings.warn("Missing BLOSC: no compression will be used.")
    COMPRESSION = tables.Filters()


def _save_level(handler, group, level, name=None):
    if isinstance(level, dict):
        # First create a new group
        new_group = handler.create_group(group, name,
                                         "dict:{}".format(len(level)))
        for k, v in level.items():
            if isinstance(k, six.string_types):
                _save_level(handler, new_group, v, name=k)
            else:
                # Key is not string, so it gets a bit more complicated.
                # If the key is not a string, we will store it as a tuple instead,
                # inside a new group
                hsh = hash(k)
                if hsh < 0:
                    hname = 'm{}'.format(-hsh)
                else:
                    hname = '{}'.format(hsh)
                new_group2 = handler.create_group(new_group, '__pair_{}'.format(hname),
                                                 "keyvalue_pair")
                new_name = '__pair_{}'.format(hname)
                _save_level(handler, new_group2, k, name='key')
                _save_level(handler, new_group2, v, name='value')

                #new_name = '__keyvalue_pair_{}'.format(hash(name))
                #setattr(group._v_attrs, new_name, (name, level))
    elif isinstance(level, list):
        # Lists can contain other dictionaries and numpy arrays, so we don't
        # want to serialize them. Instead, we will store each entry as i0, i1,
        # etc.
        new_group = handler.create_group(group, name,
                                         "list:{}".format(len(level)))

        for i, entry in enumerate(level):
            level_name = 'i{}'.format(i)
            _save_level(handler, new_group, entry, name=level_name)

    elif isinstance(level, tuple):
        # Lists can contain other dictionaries and numpy arrays, so we don't
        # want to serialize them. Instead, we will store each entry as i0, i1,
        # etc.
        new_group = handler.create_group(group, name,
                                         "tuple:{}".format(len(level)))

        for i, entry in enumerate(level):
            level_name = 'i{}'.format(i)
            _save_level(handler, new_group, entry, name=level_name)

    elif isinstance(level, np.ndarray):
        atom = tables.Atom.from_dtype(level.dtype)
        node = handler.create_carray(group, name, atom=atom, shape=level.shape,
                                     chunkshape=level.shape,
                                     filters=COMPRESSION)
        node[:] = level

    elif isinstance(level, ATTR_TYPES):
        setattr(group._v_attrs, name, level)

    elif level is None:
        # Store a None as an empty group
        new_group = handler.create_group(group, name, "nonetype:")

    else:
        ag.warning('(amitgroup.io.save) Pickling', level, ': '
                   'This may cause incompatiblities (for instance between '
                   'Python 2 and 3) and should ideally be avoided')
        node = handler.create_vlarray(group, name, tables.ObjectAtom())
        node.append(level)


def _load_level(level):
    if isinstance(level, tables.Group):
        dct = {}
        # Load sub-groups
        for grp in level:
            lev = _load_level(grp)
            n = grp._v_name
            # Check if it's a complicated pair or a string-value pair
            if n.startswith('__pair'):
                dct[lev['key']] = lev['value']
            else:
                dct[n] = lev

        # Load attributes
        for name in level._v_attrs._f_list():
            v = level._v_attrs[name]
            if isinstance(v, np.string_):
                v = v.decode('utf-8')
            dct[name] = v

        if level._v_title.startswith('list:'):
            N = int(level._v_title[len('list:'):])
            lst = []
            for i in range(N):
                lst.append(dct['i{}'.format(i)])
            return lst
        elif level._v_title.startswith('tuple:'):
            N = int(level._v_title[len('tuple:'):])
            lst = []
            for i in range(N):
                lst.append(dct['i{}'.format(i)])
            return tuple(lst)
        elif level._v_title.startswith('nonetype:'):
            return None
        else:
            return dct

    elif isinstance(level, tables.VLArray):
        if level.shape == (1,):
            return level[0]
        else:
            return level[:]
    elif isinstance(level, tables.Array):
        return level[:]


def save(path, data):
    """
    Save any Python structure to an HDF5 file. It is particularly suited for
    Numpy arrays. This function works similar to ``numpy.save``, except if you
    save a Python object at the top level, you do not need to issue
    ``data.flat[1]`` to retrieve it from inside a Numpy array of type
    ``object``.

    Four types of objects get saved natively in HDF5, the rest get serialized
    automatically.  For most needs, you should be able to stick to the four,
    which are:

    * Dictionaries
    * Lists and tuples
    * Basic data types (including strings and None)
    * Numpy arrays

    A recommendation is to always convert your data to using only these four
    ingredients. That way your data will always be retrievable by any HDF5
    reader. A class that helps you with this is `amitgroup.util.Saveable`.

    This function requires the [PyTables] module to be installed.

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

    h5file = tables.open_file(path, mode='w')
    group = h5file.root
    _save_level(h5file, group, data, name='top')
    h5file.close()


def load(path):
    """
    Loads an HDF5 saved with `save`.

    This function requires the [PyTables] module to be installed.

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

    h5file = tables.open_file(path, mode='r')
    data = _load_level(h5file.root)['top']
    h5file.close()
    return data
