"""

test_cosmology.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Mar 28 16:34:18 2013

Description: 

"""

import rt1d
import pylab as pl
import numpy as np

sim = rt1d.run.Simulation(pf={'problem_type':-1,
    'initial_redshift': 1e3, 'initial_ionization': [0.049]})
sim.run()

anl = rt1d.analyze.Simulation(sim.checkpoints)

z = np.linspace(10, 1e3)
pl.loglog(z, sim.grid.cosm.TCMB(sim.grid.cosm.zdec) * \
    (1. + z)**2 / (1. + sim.grid.cosm.zdec)**2, color = 'k')
pl.loglog(z, sim.grid.cosm.TCMB(z), color = 'k', ls = ':')

t, z, T = anl.CellTimeEvolution(field='T')
pl.loglog(z, T, color='b')

pl.xlabel(r'$z$')
pl.ylabel(r'$T_K$')
pl.xlim(10, sim.grid.zi)
pl.ylim(1, 3e3)

raw_input('')



