Physics Parameters
==================


radiative_transfer
    0) No astrophysical sources.
    1) Includes ionization/heating from astrophysical sources.

compton_scattering
    0) OFF
    1) Include Compton scattering between free electrons and CMB.

secondary_ionization
    0) All photo-electron energy deposited as heat.
    1) Compute using fits of Shull & vanSteenberg (1985).
    2) Compute using energy-dependent fits of Ricotti et al. (2002).
    3) Compute using look-up tables of Furlanetto & Stoever (2010).

clumping_factor
    Multiplicative enhancement to the recombination rate.

approx_helium
    0) Neglect helium completely.
    1) Assume singly-ionized helium fraction is equal to the hydrogen ionized fraction.

approx_sigma
    0) Compute bound-free absorption cross sections via fits of Verner et al. (1996).
    1) Approximate cross-sections as :math:`\sigma \propto \nu^{-3}`

approx_Salpha
    0) Not implemented
    1) Assume :math:`S_{\alpha} = 1`
    2) Use formulae of Chuzhoy, Alvarez, & Shapiro (2005).
    3) Use formulae of Furlanetto & Pritchard (2006)
    
nmax
    Default: 23
    
lya_injected
    Include photons injected at line-center?
    
lya_continuum
    Include photons redshifting into the red-wing of the Lyman-:math:`\alpha` line?
    
    