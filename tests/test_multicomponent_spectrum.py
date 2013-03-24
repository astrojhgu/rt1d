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
 'source_type': 'bh',
 'source_mass': 1e2,
 'source_rmax': 1e4,
 'source_evolving': 0,
 'spectrum_type': ['mcd', 'pl'],
 'spectrum_fraction': [0.5, 0.5],
 'spectrum_alpha': [None, 1.2],
 'spectrum_Emin': [13.6, 1e2],
 'spectrum_Emax': [1e4, 1e4],
 'spectrum_fcol': [1, None],
 'spectrum_logN': [0., 20.],
 'initialize_only': 2,
}

sim = rt1d.run.RT(pf = pf)
sim.rs.create_integral_table()

#sim.rs.PlotSpectrum()
#sim.rs.PlotSpectrum(t=10*rt1d.physics.Constants.s_per_myr, color='b')
#sim.rs.PlotSpectrum(t=50*rt1d.physics.Constants.s_per_myr, color='r')
#raw_input('')
