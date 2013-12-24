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

rmax = 1e2
mass = 10.
f_scatter = [0.01, 0.1, 0.5, 1.]
gamma = [-2.5, -1.5]
Emin = 10

simpl = \
{
    'source_type': 'bh', 
    'source_mass': mass,
    'source_rmax': rmax,
    'spectrum_type': 'simpl',
    'spectrum_Emin':Emin,
    'spectrum_Emax':5e4,
    'spectrum_alpha': -1.5,
    'spectrum_fsc': 0.5,
    'spectrum_uponly': True,
}

mcd = \
{
    'source_type': 'bh', 
    'source_mass': mass,
    'source_rmax': rmax,
    'spectrum_type': 'mcd',
    'spectrum_Emin': Emin,
    'spectrum_Emax': 5e4,
}

bh_mcd = rt1d.sources.RadiationSource(init_tabs=False, **mcd)
bh1 = rt1d.analyze.Source(bh_mcd)
ax = bh1.PlotSpectrum(color='k')
    
ls = ['-', '--', ':']
colors = ['b', 'g', 'r', 'm']
for i, fsc in enumerate(f_scatter):
    simpl.update({'spectrum_fsc': fsc})
    for j, alpha in enumerate(gamma):
        simpl.update({'spectrum_alpha': alpha})
        
        bh_simpl = rt1d.sources.RadiationSource(init_tabs=False, **simpl)
        bh2 = rt1d.analyze.Source(bh_simpl)
        
        if j == 0:
            label = r'$f_{\mathrm{sc}} = %g$' % fsc
        else:
            label = None
            
        ax = bh2.PlotSpectrum(color=colors[i], ls=ls[j], label=label)
        pl.draw()
        
ax.legend(loc='lower left')
ax.set_ylim(1e-8, 1e-3)
ax.set_xlim(1e2, 6e4)
pl.draw()

pl.savefig('simpl_fsc_gamma.png')
pl.savefig('simpl_fsc_gamma.eps')


