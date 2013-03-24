"""

test_multisource.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sun Mar 24 00:24:24 2013

Description: 

"""

import rt1d
import numpy as np
import pylab as pl

src1 = {'type': 'bb', 'Emin': 13.6, 'Emax': 1e2}
src2 = {'type': 'pl', 'Emin': 1e2, 'Emax': 1e3, 'alpha': 1.5}

pf = \
{
 'problem_type': 2,
 'stop_time': 10,
 'source_type': ['bb', 'bh'],
 'source_mass': [None, 1e3],
 'source_rmax': [None, 1e4],
 'source_temperature': [1e5, None],
 'spectrum_pars': [src1, src2],
}

sim = rt1d.run.Simulation(pf=pf, init_tabs=False)




