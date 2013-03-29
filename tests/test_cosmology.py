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

sim = rt1d.run.Simulation(pf={'problem_type':-1, 'dtDataDump': 1,
    'initial_redshift': 145})
sim.run()

anl = rt1d.analyze.Simulation(sim.checkpoints)

z = np.linspace(10, 145)
pl.loglog(z, sim.grid.cosm.nH0 * (1. + z)**3, color='k')

t, z, n = anl.CellTimeEvolution(field='n')
pl.scatter(z, n, color='b', facecolors='none', s=50)

pl.xlabel(r'$z$')
pl.ylabel(r'$n_{\mathrm{H}} \ \mathrm{cm}^{-3}$')

raw_input('')
pl.clf()

z = np.linspace(10, 145)
pl.loglog(z, sim.grid.cosm.TCMB(sim.grid.cosm.zdec) * \
    (1. + z)**2 / (1. + sim.grid.cosm.zdec)**2, color = 'k')

t, z, T = anl.CellTimeEvolution(field='T')
pl.scatter(z, T, color='b', facecolors='none', s=50)

pl.xlabel(r'$z$')
pl.ylabel(r'$T_K$')

raw_input('')

