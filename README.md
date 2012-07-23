Amit Group
==========

Code For Yali Amit's Research Group written in Python.

Requirements
------------

 * [Python](http://python.org/) (2.6>=)
 * [Cython](https://github.com/cython/cython) (0.16>=)
 * [Numpy](https://github.com/numpy/numpy)
 * [Scipy](https://github.com/scipy/scipy)

Installation
------------

    [sudo] make install 

### For developers

If you are planning to develop for `amitgroup`, it might be nicer to run it in place. For this, compile the Cython-driven code by 

    make inplace 

and then of course don't forget to add the top `amitgroup` directory to your `PYTHONPATH`.

### OS X

On some more recent versions of OS X, the default compiler is clang and not gcc. If you have problems compiling using clang (a problem documented on OS X 10.8) you can try

    export CC=gcc
    python setup.py install 

It's only OpenMP that clang does not support, so currently this should not be a problem, since no code uses that yet.
