.. _installation:

Getting started
===============

This explains how to install the package ``amitgroup``.

Requirements
------------

 * Python_ (2.6>=)
 * Cython_ (0.16>=)
 * Numpy_ 
 * Scipy_

Select features also require

 * matplotlib_
 * pygame_

Installation
------------

If you are a regular user, download and install it by running::

    $ git clone git@github.com:amitgroup/amitgroup.git
    $ cd amitgroup
    $ make install 

Example
-------

Now, to see if it was successfully installed, fire up a Python prompt and type::

    >>> import amitgroup as ag

If that works, you can try for instance::

    >>> ag.util.zeropad(np.ones((3, 1), dtype=int), (1, 2)) 
    array([[0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0]])

Advanced installation 
---------------------


For developers of amitgroup
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are planning to develop for `amitgroup`, it might be nicer to run it in place. For this, compile the Cython-driven code by::

    $ make inplace 

Don't forget to add the top ``amitgroup`` directory to your ``PYTHONPATH``. On most Linux/OS X systems, this is done by opening the file ``~/.bashrc`` and adding the following::


    export PYTHONPATH=/path/to/your/amitgroup:$PYTHONPATH

If you do not have a ``~/.bashrc``, then look for a ``~/.bash_profile`` or a ``~/.profile`` and add the same line. 

OS X
~~~~

On some more recent versions of OS X, the default compiler is clang and not gcc. If you have problems compiling using clang (a problem documented on OS X 10.8) you can try::

    $ export CC=gcc
    $ python setup.py install 

It's only OpenMP that clang does not support, so currently this should not be a problem, since no code uses that yet.

.. _Python: http://python.org/
.. _Cython: https://github.com/cython/cython
.. _Numpy: https://github.com/numpy/numpy
.. _Scipy: https://github.com/scipy/scipy
.. _matplotlib: http://matplotlib.sourceforge.net
.. _pygame: http://www.pygame.org/
