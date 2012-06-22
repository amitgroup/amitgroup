Amit Group
==========

Code For Yali Amit's Research Group written in Python.

Requirements
------------

 * [Cython](https://github.com/cython/cython) (0.16>=)
 * [Numpy](https://github.com/numpy/numpy)
 * [Scipy](https://github.com/scipy/scipy)

Installation
------------

    [sudo] python setup.py install

If you want to compile the Cython-driven code in place, without installing it to your site-packages folder, run 

    python setup.py build_ext --inplace

### OS X

On some more recent versions of OS X, the default compiler is clang and not gcc. If you have problems compiling using clang (a problem documented on OS X 10.8) you can try

    export CC=gcc
    python setup.py install 

