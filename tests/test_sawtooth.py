"""

test_sawtooth.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Aug 16 19:28:47 MDT 2013

Description: Or rather, ``picket fence,'' i.e. Ahn et al. (2009).

"""

import rt1d
import matplotlib.pyplot as pl
import numpy as np
from math import ceil
from scipy.integrate import quad
from rt1d.physics.Constants import E_LyA, E_LL, cm_per_kpc, erg_per_ev, \
    ev_per_hz

pars = {
    'source_type': 'star',
    'source_temperature': 1e5,
    'spectrum_type': 'bb',
    'spectrum_Emin': E_LyA,
    'spectrum_Emax': E_LL,
    'spectrum_EminNorm': 0.1,
    'spectrum_EmaxNorm': 5e2,
}

rs = rt1d.sources.RadiationSource(**pars)
Qlw = rs.Lbol * quad(lambda E: rs.Spectrum(E) / E, E_LyA, E_LL)[0]

r_kpc = np.logspace(0, 3, 100)      # kpc
r_cm = r_kpc * cm_per_kpc

# Flux without redshift photons, pure inverse square law
pl.loglog(r_kpc, Qlw / 4. / np.pi / r_cm**2, color='k')

pl.xlabel(r'$r \ (\mathrm{kpc})$')
pl.ylabel('Flux')

def J(r, E, nmax=23):
    """
    Compute specific flux at a given radius assuming the luminosity
    and SED are time-independent.    
    
    """
    
    # Closest Lyman line (from above)
    n = ceil(np.sqrt(E_LL / (E_LL - E)))
    
    if n > nmax:
        return 0.0
    
    En =  E_LL * (1. - 1. / n**2)
    #zmax = En * (1. + z) / E - 1.
    
    return rs.Lbol * rs.Spectrum(En) * ev_per_hz / En / erg_per_ev / 4. / np.pi / r**2
    
#def Q(r, nmax=23):
#    return quad(lambda E: J(r, E, nmax=nmax) / E, E_LyA, E_LL)[0]
    
# Flux without redshift photons, pure inverse square law
pl.loglog(r_kpc, map(lambda r: J(r, 10.21), r_cm), color='k', ls='--')
    
    