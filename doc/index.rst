.. rt1d documentation master file, created by
   sphinx-quickstart on Sat Jul  6 17:37:34 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

rt1d
==============================
`rt1d <https://bitbucket.org/mirochaj/rt1d>`_ is a 1D radiative transfer code
developed to study the ionization and thermal evolution of gas in the vicinity
of stars, accreting black holes, or really any source of ultraviolet and/or
X-ray photons you can think of. It was also designed to facilitate detailed
studies of numerical effects that may arise in radiative transfer simulations.

A paper including some discussion of its inner-workings can be found
`here <http://adsabs.harvard.edu/abs/2012ApJ...756...94M>`_.

Currently, rt1d has support for the following:
    * Standard non-equilibrium chemistry solver for hydrogen and helium.
    * Continuous or discrete SEDs, choice of using multi-group or
      multi-frequency methods for discrete SEDs.
    * Photon-conserving and non-photon-conserving algorithms.
    * Secondary ionization and heating rates from Shull & vanSteenberg (1985), Ricotti,
      Gnedin, & Shull (2002), or Furlanetto & Stoever (2010).
    * Cosmological expansion.
    * Time-dependent ionizing backgrounds.
    * Optimization of multi-frequency SEDs using method of `Mirocha et al. (2012) <http://adsabs.harvard.edu/abs/2012ApJ...756...94M>`_.
.. * Time-dependent source luminosities and SEDs.
.. * Infinite speed-of-light approximation or explicit treatment of the speed-of-light.
    

Contents
--------
.. toctree::
   :maxdepth: 3
   
   Home <self>
   install
   structure
   examples
   parameters
   analysis