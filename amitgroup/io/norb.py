from __future__ import division, print_function, absolute_import

import os
import struct
import numpy as np


def load_small_norb(dataset, norb_path=None):
    assert dataset in ('training', 'testing')

    if norb_path is None:
        norb_path = os.environ['NORB_DIR']

    name = {
        'training': 'smallnorb-5x46789x9x18x6x2x96x96-training',
        'testing': 'smallnorb-5x01235x9x18x6x2x96x96-testing',
    }[dataset]

    dat_fn = os.path.join(norb_path, name + '-dat.mat')
            
    with open(dat_fn, 'rb') as f:
        byte_matrix = struct.unpack('<I', f.read(4))[0]
        assert byte_matrix == 0x1e3d4c55, "Only supports byte matrix for now"

        ndim = struct.unpack('<I', f.read(4))[0]
        dims = struct.unpack('<IIII', f.read(4 * 4))

        X = np.fromfile(f, dtype=np.uint8, count=np.prod(dims)).reshape(dims)

    cat_fn = os.path.join(norb_path, name + '-cat.mat')

    with open(cat_fn, 'rb') as f:
        byte_matrix = struct.unpack('<I', f.read(4))[0]
        assert byte_matrix == 0x1e3d4c54, "Only supports intger matrix for now"

        ndim = struct.unpack('<I', f.read(4))[0]
        dim0 = struct.unpack('<III', f.read(4 * 3))[0]

        y = np.fromfile(f, dtype=np.int32, count=dim0)

    return X, y

