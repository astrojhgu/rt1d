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

colors = ['c', 'm', 'r', 'g', 'b', 'k']

mp = None
for i, res in enumerate([2**n for n in np.arange(5, 10)]):
    sim = rt1d.run.RT(pf = {'problem_type': 1, 'grid_cells': res})
    anl = rt1d.analysis.Analyze(sim.checkpoints)

    plot_anl = plot_sol = True
    if res < 1024:
        plot_anl = plot_sol = False
    
    mp = anl.PlotIonizationFrontEvolution(color = colors[i], mp = mp, 
        anl = plot_anl, label = r'$\Delta x = 1 / %i$' % res,
        plot_solution = plot_sol)
        
    pl.draw()    # seems a bit redundant
        
    del anl

mp.fix_ticks()    
pl.rcParams['legend.fontsize'] = 14
mp.grid[1].legend(loc = 'lower right', frameon = False, ncol = 2)
pl.savefig('test_convergence.png')
raw_input('')
    