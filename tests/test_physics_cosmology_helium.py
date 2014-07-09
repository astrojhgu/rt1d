"""

test_physics_cosmology_helium.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Mar 28 16:34:18 2013

Description: Recombination test problem - include helium.

"""

import rt1d
import matplotlib.pyplot as pl
import numpy as np
from multiplot import multipanel

sim = rt1d.run.Simulation(pf={'problem_type':-1,
    'initial_redshift': 1e3, 'initial_ionization': [0.049, 0.049],
    'Z':[1,2], 'abundance':[1.0,0.08]})
sim.run()

anl = rt1d.analyze.Simulation(sim.checkpoints)

z = np.linspace(10, 1e3)

mp = multipanel(dims=(2,1), useAxesGrid=False, share_all=False)

mp.grid[0].loglog(z, sim.grid.cosm.TCMB(sim.grid.cosm.zdec) * \
    (1. + z)**2 / (1. + sim.grid.cosm.zdec)**2, color = 'k',
    label=r'analytic')
mp.grid[0].loglog(z, sim.grid.cosm.TCMB(z), color = 'k', ls = ':')

z1, T = anl.CellEvolution(field='Tk', redshift=True)
mp.grid[0].loglog(z1, T, color='b', label='rt1d')
z2, Ts = anl.CellEvolution(field='Ts', redshift=True)
mp.grid[0].loglog(z2, Ts, color='b', ls = '--')

mp.grid[0].legend(frameon=False, loc='lower right')   

# Ionization
z3, xHII = anl.CellEvolution(field='h_2', redshift=True)
z4, xHeII = anl.CellEvolution(field='he_2', redshift=True)
mp.grid[1].loglog(z3, xHII, color='k', ls = '-', label=r'$x_{\mathrm{HII}}$')
mp.grid[1].loglog(z4, xHeII, color='k', ls = '--', label=r'$x_{\mathrm{HeII}}$')
mp.grid[1].legend(loc='upper left', frameon=False, ncol=2)

mp.grid[1].set_xlabel(r'$z$')
mp.grid[0].set_ylabel(r'$T_K$')
mp.grid[1].set_ylabel(r'$x_i$')
mp.grid[0].set_xlim(10, sim.grid.zi)
mp.grid[1].set_xlim(10, sim.grid.zi)
mp.grid[0].set_ylim(1, 2e3)
mp.grid[1].set_ylim(1e-6, 1)
mp.fix_ticks()   
raw_input('')    



