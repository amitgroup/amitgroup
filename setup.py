#!/usr/bin/env python

from distutils.core import setup

setup(name='amitgroup',
    version='0',
    url="https://github.com/amitgroup/amitgroup",
    description="Code for Yali Amit's Research Group",
    packages=[
        'amitgroup', 
        'amitgroup.features',
        'amitgroup.io',
        'amitgroup.stats'
    ],
)
