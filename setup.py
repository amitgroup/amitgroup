#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

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
        Extension("amitgroup.features.bedges", ["amitgroup/features/bedges.pyx"], extra_compile_args=['-fopenmp'],extra_link_args=['-fopenmp']),
    ]
)
