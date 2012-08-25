.. _wavelet2d:

.. currentmodule:: amitgroup.util.wavelet

========================
Wavelet Transforms in 2D
========================

We provide 2D periodic Daubechies wavelet transforms as a faster alternative to for instance pyWavelets_, especially if you plan to do repeated transforms on the same size input.

With our approach, you first create the transform function, where you get to specify the size of the input data, the wavelet type and how many coefficient levels that you plan to use. 

Usage
-----
Wavelets transforms are done by first creating the transform function using one of the following factory functions. Further instructions and examples can be seen by clicking the links below:

.. autosummary:: 
   :toctree: generated/
     
   wavedec2_factory
   waverec2_factory

Speed improvements
------------------
The speed has been improved by several means. Mainly faster storage, pre-processing matrix filters and the option of not computing coefficients that you don't plan to use anyway. 

Faster storage
~~~~~~~~~~~~~~

PyWavelets_ returns a list of tuples of numpy arrays, for 2D wavelets. This takes a lot of Pythonic operations and is thus very slow. In addition to that, if you need all the coefficients in a contiguous block of memory, your incur additional conversion costs. We support only periodic wavelets, which means that each level of coefficients have 3 arrays of sides of powers of two. All these levels can fit like Russian dolls into a matrix of the same size as the input data. For instance, given an input of size 8x8, the coefficients will be returned stored as following in PyWavelets_, which is analgous to the MATLAB Wavelet Toolbox::

    [cA3, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1)]

But instead, in our code it is stored in a 8 x 8 array with the following layout::

    -----------------------------------------------------------------
    |  cA3  |  cH3  |               |                               |
    | (1x1) | (1x1) |      cH2      |                               |
    -----------------     (2x2)     |                               |
    |  cV3  |  cD3  |               |                               |
    | (1x1) | (1x1) |               |              cH1              |
    ---------------------------------             (4x4)             |
    |               |               |                               |
    |      cV2      |      cD2      |                               |
    |     (2x2)     |     (2x2)     |                               |
    |               |               |                               |
    |               |               |                               |
    -----------------------------------------------------------------
    |                               |                               |
    |                               |                               |
    |                               |                               |
    |                               |                               |
    |              cV1              |              cD1              |
    |             (4x4)             |             (4x4)             |
    |                               |                               |
    |                               |                               |
    |                               |                               |
    |                               |                               |
    |                               |                               |
    -----------------------------------------------------------------

Throw-away coefficients
~~~~~~~~~~~~~~~~~~~~~~~

Another speed improvement can be made if you plan to throw away some of the highest frequency coefficients. For instance, if you know that in the above 8x8 transform, you know you do not need any of the 4x4 coefficients, you can specify that as following::

    >>> wavedec2 = ag.util.wavelet.wavedec2_factory((8, 8), wavelet='db2', levels=2)
    >>> coefs = wavedec2(np.ones((8, 8)))
    >>> print coefs.shape
    (4, 4)

We are able to offer these speed improvements by skipping the high-pass filters for the unnecessary coefficient levels, and combining several low-pass filters into one. The filtering is implemented using matrix multiplication, and the matrices (filters) needed are precomputed in the factory functions. For high-dimensional matrices, we take advantage of SciPy's sparse matrix functionality.

Coefficient conversions
-----------------------

.. note::

    These conversions will hurt your performance, and should be avoided for performance-sensitive code.

The coefficients arrays can be converted to and from the same layout as pyWavelets_, which is great for comparison or migration:

.. autosummary:: 
   :toctree: generated/
     
   pywt2array

If you ever need to plot the coefficient on one axis, it can be good to have a flattened array, that is sorted in terms of the different levels of coefficients (notice that simply running ``flatten()`` on coefficients will not achieve this):

.. autosummary:: 
   :toctree: generated/
     
   smart_flatten
   smart_deflatten 

Benchmarks
----------

PyWavelets_ is a great library with a rich feature set. However, even though it is largely Cython-powered, the performance can be lacking, especially for 2D transforms (where my profiler tells me that a lot of time is spent shuffling memory, such as running ``transpose()``), and especially if you are running small-image transforms, but *a lot* of them:

.. image:: images/wavelet_benchmark.png 

.. _pyWavelets: http://www.pybytes.com/pywavelets/ 
