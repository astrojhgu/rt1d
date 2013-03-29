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

sim = rt1d.run.Simulation(pf={'problem_type':-1, 'dtDataDump': 0.1})
sim.run()

anl = rt1d.analyze.Simulation(sim.checkpoints)

z = np.linspace(10, 400)
pl.loglog(z, sim.grid.cosm.nH0 * (1. + z)**3, color = 'k')

t, z, n = anl.CellTimeEvolution(field='n')
pl.loglog(z, n)
t, z, de = anl.CellTimeEvolution(field='de')
pl.loglog(z, de)


raw_input('')

