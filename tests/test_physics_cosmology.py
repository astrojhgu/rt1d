"""

test_physics_cosmology.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Mar 28 16:34:18 2013

Description: 

"""

import rt1d
import numpy as np
import matplotlib.pyplot as pl

sim = rt1d.run.Simulation(pf={'problem_type':-1,
    'initial_redshift': 3e3, 'initial_ionization': [0.99999],
    'rate_source': 'chianti'})
sim.run()

anl = rt1d.analyze.Simulation(sim.checkpoints)

fig1 = pl.figure(1)
fig2 = pl.figure(2)

ax1 = fig1.add_subplot(111)
ax2 = fig2.add_subplot(111)

z = np.linspace(10, 1e3)
ax1.loglog(z[z <= sim.grid.cosm.zdec], 
    sim.grid.cosm.TCMB(sim.grid.cosm.zdec) * \
    (1. + z[z <= sim.grid.cosm.zdec])**2 / (1. + sim.grid.cosm.zdec)**2, 
    color='k', label=r'analytic')
ax1.loglog(z, sim.grid.cosm.TCMB(z), color = 'k', ls = ':')

z1, T = anl.CellEvolution(field='Tk', redshift=True)
ax1.loglog(z1, T, color='b', label='rt1d')
z2, Ts = anl.CellEvolution(field='Ts', redshift=True)
ax1.loglog(z2, Ts, color='b', ls = '--')

ax1.set_xlabel(r'$z$')
ax1.set_ylabel(r'$T_K$')
ax1.set_xlim(10, sim.grid.zi)
ax1.set_ylim(1, 2e3)
    
ax1.legend(frameon=False, loc='lower right')   

z, x = anl.CellEvolution(field='h_2', redshift=True)
ax2.loglog(z, x, color='k')
ax2.set_ylabel(r'$x_{\mathrm{HII}}$')
ax2.set_xlabel(r'$z$')


pl.draw()
