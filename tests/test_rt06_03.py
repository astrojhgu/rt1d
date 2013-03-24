"""

test_single_zone.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Jan 18 09:35:35 2013

Description: 

"""

import rt1d

sim = rt1d.run.RT(pf = {'problem_type': 0, 'optically_thin': 1})

anl = rt1d.analysis.Analyze(sim.checkpoints)

t, z, xHI = anl.CellTimeEvolution(field = 'h_1')
t, z, T = anl.CellTimeEvolution(field = 'T')

mp = rt1d.analysis.multiplot(dims = (2, 1), share_all = False, 
    useAxesGrid = False, panel_size = (0.5, 1))
    
s_per_yr = rt1d.physics.Constants.s_per_yr
mp.grid[0].loglog(t / s_per_yr, xHI, color = 'k')
mp.grid[1].loglog(t / s_per_yr, T, color = 'k')  

mp.grid[0].set_xlim(1e-6, 1e7)
mp.grid[1].set_xlim(1e-6, 1e7)
mp.grid[0].set_ylim(1e-8, 1.5)
mp.grid[1].set_ylim(1e2, 1e5)
    
mp.grid[0].set_ylabel(r'$x_{\mathrm{HI}}$')
mp.grid[1].set_ylabel(r'$T \ (\mathrm{K})$')
mp.grid[1].set_xlabel(r'$t \ (\mathrm{yr})$')
mp.fix_ticks()
raw_input('')
