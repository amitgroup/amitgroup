#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
import numpy as np
import os.path

import Cython
from Cython.Distutils import build_ext
import re


def cython_extension(modpath, mp=False):
    extra_compile_args = ["-O3"]
    extra_link_args = []
    if mp:
        extra_compile_args.append('-fopenmp')
        extra_link_args.append('-fopenmp')
    filepath = os.path.join(*modpath.split('.')) + ".pyx"
    return Extension(modpath, [filepath],
                     extra_compile_args=extra_compile_args,
                     extra_link_args=extra_link_args)

setup(name='amitgroup',
    cmdclass={'build_ext': build_ext},
    version='0',
    url="https://github.com/amitgroup/amitgroup",
    description="Code for Yali Amit's Research Group",
    packages=[
        'amitgroup',
        'amitgroup.features',
        'amitgroup.io',
        'amitgroup.stats',
        'amitgroup.util',
        'amitgroup.util.wavelet',
        'amitgroup.plot',
    ],
    ext_modules=[
        cython_extension("amitgroup.features.features"),
        cython_extension("amitgroup.features.spread_patches"),
        cython_extension("amitgroup.features.code_parts"),
        cython_extension("amitgroup.util.interp2d"),
        cython_extension("amitgroup.util.nn2d"),
        cython_extension("amitgroup.plot.resample"),
    ],
    include_dirs=[np.get_include()],
)
