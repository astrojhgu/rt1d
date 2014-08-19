"""

test_sed_mcd.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu May  2 10:46:44 2013

Description: Plot a simple multi-color disk accretion spectrum.

"""

import rt1d
import matplotlib.pyplot as pl
import numpy as np

pars = \
{
 'source_type': 'star', 
 'source_temperature': 1e4,
 'spectrum_type': 'bb',
 'spectrum_Emin': 1.,
 'spectrum_Emax': 1e2,
}

ls = ['--', '-']
for i, T in enumerate([1e4, 1e5]):
    pars.update({'source_temperature': T})

    src = rt1d.sources.RadiationSource(init_tabs=False, **pars)
    bh = rt1d.analyze.Source(src)
    
    ax = bh.PlotSpectrum(ls=ls[i], 
        label=r'$T_{\ast} = 10^{%i} \mathrm{K}$' % (np.log10(T)))

ax.legend(loc='lower right')
ax.set_ylim(1e-8, 1)
pl.draw()


