Amit Group
==========

Code For Yali Amit's Research Group written in Python.

Requirements
------------

 * [Python](http://python.org/) (2.6>=, 3.3>=)
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

