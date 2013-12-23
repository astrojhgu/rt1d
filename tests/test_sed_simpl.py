"""

test_sed_mcd.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu May  2 10:46:44 2013

Description: Plot a simple multi-color disk accretion spectrum.

"""

import rt1d
import pylab as pl

rmax = 1e3
mass = 10.

simpl = \
{
    'source_type': 'bh', 
    'source_mass': mass,
    'source_rmax': rmax,
    'spectrum_type': 'simpl',
    'spectrum_Emin':1e2,
    'spectrum_Emax':3e4,
    'spectrum_alpha': -1.5,
    'spectrum_fsc': 0.5,
    'uponly': False,
}

mcd = \
{
    'source_type': 'bh', 
    'source_mass': mass,
    'source_rmax': rmax,
    'spectrum_type': 'mcd',
    'spectrum_Emin': 1e2,
    'spectrum_Emax': 3e4,
}

bh_simpl = rt1d.sources.RadiationSource(init_tabs=False, **simpl)
bh_mcd = rt1d.sources.RadiationSource(init_tabs=False, **mcd)
    
bh1 = rt1d.analyze.Source(bh_simpl)
bh2 = rt1d.analyze.Source(bh_mcd)

ax = bh2.PlotSpectrum(color='k', bins=100)
ax = bh1.PlotSpectrum(color='b', bins=100)
pl.draw()
