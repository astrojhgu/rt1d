Spectrum Parameters
===================
We use :math:`I_{\nu}` to denote the SED of sources. It is proportional
to the *energy* emitted at frequency :math:`\nu`, NOT the number of photons
emitted at frequency :math:`\nu`.

We use square brackets on this page to denote the units of parameters.

``spectrum_type``
    Options:

    + ``'bb'``: blackbody
    + ``'pl'``: power-law
    + ``'mcd'``; Multi-color disk (Mitsuda et al. 1984)
    + ``'simpl'``: SIMPL Comptonization model (Steiner et al. 2009)
    + ``'qso'``: Quasar template spectrum (Sazonov et al. 2004)

``spectrum_Emin``
    Minimum photon energy to consider in radiative transfer calculation [eV]

``spectrum_Emax``
    Maximum photon energy to consider in radiative transfer calculation [eV]

``spectrum_EminNorm``
    Minimum photon energy to consider in normalization [eV]

``spectrum_EmaxNorm``
    Maximum photon energy to consider in normalization [eV]
    
.. math::

    \int_{\text{spectrum_EminNorm}}^{\text{spectrum_EminNorm}} I_{\nu} d\nu = 1

``spectrum_alpha``
    Power-law index of emission. Only used if ``spectrum_type`` is ``pl`` or ``simpl``
    Default: -1.5
    
    Defined such that    

.. math::
    
    I_{\nu} \propto \nu^{\alpha}
    
Recall that :math:`I_{\nu}` is proportional to the energy, not the number of photons,
emitted at frequency :math:`\nu`.
    
``spectrum_logN``
    Base-10 logarithm of the neutral absorbing column (hardens spectrum)
    Default: :math:`-\infty`
    
``spectrum_Rmax``
    If ``spectrum_type`` is 'mcd', this parameter sets the maximum size of the
    accretion disk being considered. [gravitational radii, :math:`R_g`]
    
``spectrum_fcol``
    Color correction factor, acts to harden BH accretion spectrum. 
    Default: 1.7
    
``spectrum_kwargs``
    A dictionary containing any (or all) spectrum parameters.
    


