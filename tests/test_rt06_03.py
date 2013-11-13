"""

test_rt06_03.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Jan 18 09:35:35 2013

Description: Single zone ionization/recombination + heating/cooling test.

"""

import rt1d
from multiplot import multipanel

sim = rt1d.run.Simulation(problem_type=0, optically_thin=1)
sim.run()

anl = rt1d.analyze.Simulation(sim.checkpoints)

t, xHI = anl.CellEvolution(field='h_1')
t, T = anl.CellEvolution(field='Tk')

mp = multipanel(dims=(2, 1), panel_size=(1, 0.5))
    
s_per_yr = rt1d.physics.Constants.s_per_yr
mp.grid[0].loglog(t / s_per_yr, xHI, color = 'k')
mp.grid[1].loglog(t / s_per_yr, T, color = 'k')  

mp.grid[0].set_xlim(1e-6, 1e7)
mp.grid[1].set_xlim(1e-6, 1e7)
mp.grid[0].set_ylim(1e-8, 1.5)
mp.grid[1].set_ylim(1e2, 1e5)
    
mp.grid[0].set_ylabel(r'$x_{\mathrm{HI}}$')
mp.grid[1].set_ylabel(r'$T \ (\mathrm{K})$')
mp.grid[0].set_xlabel(r'$t \ (\mathrm{yr})$')
mp.fix_ticks()

