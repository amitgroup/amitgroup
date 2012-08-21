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

Machine Learning (:mod:`amitgroup.ml`)
--------------------------------------
.. currentmodule:: amitgroup.ml

.. autosummary::
   :toctree: generated/

   bernoulli_deformation
   image_deformation

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
