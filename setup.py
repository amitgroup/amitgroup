#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os.path

def CythonExtension(modpath, mp=False):
    extra_compile_args = []
    extra_link_args = []
    if mp:
        extra_compile_args.append('-fopenmp')
        extra_link_args.append('-fopenmp') 
    filepath = os.path.join(*modpath.split('.')) + ".pyx"
    return Extension(modpath, [filepath], extra_compile_args=extra_compile_args, extra_link_args=extra_link_args)

setup(name='amitgroup',
    cmdclass = {'build_ext': build_ext},
    version='0',
    url="https://github.com/amitgroup/amitgroup",
    description="Code for Yali Amit's Research Group",
    packages=[
        'amitgroup', 
        'amitgroup.features',
        'amitgroup.io',
        'amitgroup.stats'
    ],
    ext_modules = [
        CythonExtension("amitgroup.features.bedges", mp=True),
        CythonExtension("amitgroup.math.interp2d"),
        CythonExtension("amitgroup.ml.aux"),
    ]
)
