.. rt1d documentation master file, created by
   sphinx-quickstart on Sat Jul  6 17:37:34 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

rt1d
==============================
rt1d is a 1D radiative transfer code developed to study the ionization and
thermal evolution of gas in the vicinity of stars, accreting black holes, or
really any source of ultraviolet and/or X-ray photons you can think of. It was
also designed to facilitate detailed studies of numerical effects that may
arise in radiative transfer simulations.

A paper including some discussion of its inner-workings can be found
`here <http://adsabs.harvard.edu/abs/2012ApJ...756...94M>`_.

Physics Available
-----------------
* Secondary ionization and heating via Shull & vanSteenberg (1985), Ricotti,
  Gnedin, & Shull (2002), or Furlanetto & Stoever (2010).
* Time-dependent source luminosity and/or SED.
* Continuous or discrete SEDs, choice to of using multi-group or
  multi-frequency methods for discrete calculations.
* Choice between photon-conserving or non-photon-conserving algorithms.
* Choice to treat speed-of-light explicitly.
* Can use `dengo <https://bitbucket.org/MatthewTurk/dengo>`_ package to create
  advanced chemical networks, though they are not yet coupled to the
  radiation field.

Topics
------
.. toctree::
   :maxdepth: 3
   
   Home <self>
   install
   getting_started
   examples