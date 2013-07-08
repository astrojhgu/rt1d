"""

test_multi_component.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sat Mar 23 17:17:16 2013

Description: Demonstrate how to construct a multi-component SED.
Here, we use a multi-color disk + power-law spectrum often 
used to model emission from the vicinity of an accreting black hole.

"""

import rt1d
import numpy as np
import pylab as pl

pf = \
{
 'problem_type': 2,
 'stop_time': 10,
 'source_type': 'bh',
 'source_mass': 1e3,
 'source_rmax': 1e3,
 'source_evolving': 0,
 'spectrum_type': ['mcd', 'pl'],
 'spectrum_fraction': [0.5, 0.5],
 'spectrum_alpha': [None, -1.2],
 'spectrum_Emin': [13.6, 1e2],
 'spectrum_Emax': [1e4, 1e4],
 'spectrum_fcol': [1, None],
 'spectrum_logN': [0., 20.], 
}

sim = rt1d.run.Simulation(pf=pf)
sim.run()

anl = rt1d.analyze.Simulation(sim.checkpoints)

anl.IonizationProfile(t=[1, 5, 10], color='k')
raw_input('')
pl.close()

anl.TemperatureProfile(t=[1, 5, 10], color='k')
raw_input('')
pl.close()
