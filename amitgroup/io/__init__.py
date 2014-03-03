from __future__ import absolute_import

from .mnist import load_mnist
from .book import *
from .examples import load_example

try:
    import tables
    _pytables_ok = True 
except ImportError:
    _pytables_ok = False

if _pytables_ok:
    from .hdf5io import load, save
else:
    def _f(*args, **kwargs):
        raise ImportError("You need PyTables for this function") 
    load = save = _f
