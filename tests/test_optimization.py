"""

test_optimization.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Feb 25 09:27:17 2013

Description: 

"""

import rt1d, ndmin
import numpy as np
import pylab as pl

erg_per_ev = rt1d.physics.Constants.erg_per_ev 

Z = [1]
logN = [np.linspace(15, 20, 26)]

src = {'problem_type': 2}
sedop = rt1d.run.Optimization(logN=logN, Z=Z, nfreq=4, rs=src)
sedop(1e4)

#Eopt, LEopt = np.array([17.98, 31.15, 49.09, 76.98]), np.array([0.23, 0.36, 0.24, 0.06])
Eopt, LEopt = ndmin.util.split_list(sedop.sampler.best_xarr)

best_phi = sedop.discrete_tabs(Eopt, LEopt)['logPhi_h_1']

pl.loglog(10**sedop.logN[0], 10**sedop.rs.tabs['logPhi_h_1'], color = 'k')
pl.loglog(10**sedop.logN[0], 10**best_phi / erg_per_ev, color = 'b')

raw_input('')    

