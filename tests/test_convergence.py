"""

test_convergence.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Jan 16 14:46:46 2013

Description: Vary grid resolution for RT06 #1.

"""

import rt1d
import pylab as pl
import numpy as np

maxres = 256
colors = ['c', 'm', 'r', 'g', 'b', 'k']

mp = None
for i, res in enumerate([2**n for n in np.arange(5, 10)]):
    if res > maxres:
        break
    
    sim = rt1d.run.Simulation(problem_type=1, grid_cells=res)
    sim.run()
    
    anl = rt1d.analyze.Simulation(sim.checkpoints)

    plot_anl = plot_sol = True
    if res < maxres:
        plot_anl = plot_sol = False
    
    mp = anl.PlotIonizationFrontEvolution(color=colors[i], mp=mp, 
        anl=plot_anl, label=r'$\Delta x = 1 / %i$' % res,
        plot_solution=plot_sol)
        
    pl.draw()    # seems a bit redundant
        
    del anl

mp.fix_ticks()    
pl.rcParams['legend.fontsize'] = 14
mp.grid[0].legend(loc='lower right', frameon=False, ncol=2)

    