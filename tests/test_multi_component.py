"""

test_multicomponent_spectrum.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sat Mar 23 17:17:16 2013

Description: 

"""

import rt1d
import numpy as np

pf = \
{
 'problem_type': 2,
 'stop_time': 10,
 'source_type': 'bh',
 'source_mass': 1e3,
 'source_rmax': 1e4,
 'source_evolving': 1,
 'spectrum_type': ['mcd', 'pl'],
 'spectrum_fraction': [0.5, 0.5],
 'spectrum_alpha': [None, 1.2],
 'spectrum_Emin': [13.6, 1e2],
 'spectrum_Emax': [1e4, 1e4],
 'spectrum_fcol': [1, None],
 'spectrum_logN': [0., 20.],
}

pf2 = pf.copy()
pf2['source_evolving'] = 0

sim = rt1d.run.Simulation(pf=pf)
sim.run()

sim2 = rt1d.run.Simulation(pf=pf2)
sim2.run()

ax = sim.IonizationProfile(t=[1, 5, 10], color='k')
sim2.IonizationProfile(ax=ax, t=[1, 5, 10], color='k')

raw_input('')
