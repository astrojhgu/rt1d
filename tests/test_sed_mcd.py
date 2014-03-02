"""

test_sed_mcd.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu May  2 10:46:44 2013

Description: Plot a simple multi-color disk accretion spectrum.

"""

import rt1d
import pylab as pl
import numpy as np

bh_pars = \
{
    'source_type': 'bh', 
    'source_mass': 10.,
    'source_rmax': 1e3,
    'spectrum_type': 'mcd',
    'spectrum_Emin':10.,
    'spectrum_Emax':1e4,
    'spectrum_logN': 18.,
}

src = rt1d.sources.RadiationSource(init_tabs=False, **bh_pars)
bh = rt1d.analyze.Source(src)

ax = bh.PlotSpectrum(ax=ax)
pl.draw()


