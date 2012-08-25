.. _reference:

API Reference
=============

.. currentmodule:: amitgroup

.. _features:

Feature extraction (:mod:`amitgroup.features`)
----------------------------------------------
.. currentmodule:: amitgroup.features

.. autosummary:: 
   :toctree: generated/

   bedges

.. _io:

Input/Output (:mod:`amitgroup.io`)
----------------------------------
.. currentmodule:: amitgroup.io

.. autosummary::
   :toctree: generated/

   load_all_images
   load_example
   load_image
   load_mnist

Statistics (:mod:`amitgroup.stats`)
-----------------------------------
.. currentmodule:: amitgroup.stats

.. autosummary::
   :toctree: generated/

   BernoulliMixture
   bernoulli_deformation
   image_deformation

Utilities (:mod:`amitgroup.util`)
---------------------------------
.. currentmodule:: amitgroup.util

.. autosummary::
   :toctree: generated/

   DisplacementField
   DisplacementFieldWavelet
   blur_image
   interp2d
   zeropad

Wavelets (:mod:`amitgroup.util.wavelet`)
----------------------------------------
.. currentmodule:: amitgroup.util.wavelet

.. autosummary::
   :toctree: generated/
  
   wavedec2_factory
   waverec2_factory
   smart_flatten
   smart_deflatten
   pywt2array

Plotting (:mod:`amitgroup.plot`)
--------------------------------
.. currentmodule:: amitgroup.plot

The plotting module adds some convenience functions for using matplotlib_ and pygame_.

.. autosummary::
   :toctree: generated/
 
   PlottingWindow
   deformation
   images

.. _matplotlib: http://matplotlib.sourceforge.net
.. _pygame: http://www.pygame.org/
