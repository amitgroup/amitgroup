from __future__ import division, print_function, absolute_import 
import amitgroup as ag
import numpy as np
import os


def load_cifar_10(section, offset=0, count=10000):
    assert section in ['training', 'testing']

    # TODO: This loads it from hdf5 files that I have prepared.

    # For now, only batches that won't be from several batches
    assert count <= 10000
    assert offset % count == 0
    assert 10000 % count == 0

    batch_offset = 0

    if section == 'training':
        batch_number = offset // 10000 + 1
        assert 1 <= batch_number <= 5
        name = 'cifar_{}.h5'.format(batch_number)
        batch_offset = offset % 10000
    else:
        name = 'cifar_test.h5'

    data = ag.io.load(os.path.join(os.environ['CIFAR10_DIR'], name))

    X = data['data']
    y = data['labels']

    # TODO: This has to read the whole batch
    return (X[batch_offset:batch_offset+count].reshape(-1, 3, 32, 32),
            y[batch_offset:batch_offset+count])
