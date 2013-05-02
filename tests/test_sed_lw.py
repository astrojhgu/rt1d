"""

test_lyman_alpha.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Apr 11 10:03:21 2013

Description: 

"""

import rt1d
import pylab as pl
from rt1d.physics.Constants import *

bb_pars = {
           'source_type': 'star', 
           'spectrum_type': 'bb',
           'spectrum_Emin':E_LyA,
           'spectrum_Emax':1e2,
           "spectrum_EminNorm": 1e-1,
           "spectrum_EmaxNorm": 1e3
          }

bb = rt1d.analyze.Source(rt1d.sources.RadiationSource(init_tabs=False, 
    **bb_pars))

pars = {
        "source_type": 'diffuse',
        "spectrum_type": ['pl', 'pl'],
        "spectrum_fraction": [1, 1],
        "spectrum_Emin": [E_LyA, E_LyB],
        "spectrum_Emax": [E_LyB, E_LL],
        "spectrum_EminNorm": [E_LyA, E_LyB],
        "spectrum_EmaxNorm": [E_LyB, E_LL],
        "spectrum_alpha": [1.29, 1.1],
       }

src = rt1d.analyze.Source(rt1d.sources.RadiationSource(init_tabs=False, 
    **pars))

ax = bb.PlotSpectrum()
ax = src.PlotSpectrum(ax=ax, color='b')

ax.set_xlim(10, 14.)
ax.set_xscale('linear')
ax.set_yscale('log')
ax.plot([E_LyB]*2, ax.get_ylim(), color='k', ls=':')
pl.draw()

raw_input('')

