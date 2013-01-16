"""

test_convergence.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Jan 16 14:46:46 2013

Description: Vary grid resolution for RT06 #1.

"""

import rt1d
import pylab as pl

colors = ['c', 'm', 'r', 'g', 'b', 'k']

anl = []
for i, res in enumerate([32, 64, 128, 256, 512, 1024]):
    sim = rt1d.run.RTsim(pf = {'problem_type': 1, 'grid_cells': res})
    anl = rt1d.analysis.Analyze(sim.checkpoints)

    if i == 0:
        mp = anl.PlotIonizationFrontEvolution(color = colors[i], 
            label = r'$\Delta x = 1 / %i$' % res)
    elif res < 1024:
        anl.PlotIonizationFrontEvolution(color = colors[i], mp = mp, 
            anl = False, label = r'$\Delta x = 1 / %i$' % res,
            plot_solution = False)
    else:
        anl.PlotIonizationFrontEvolution(color = colors[i], mp = mp, 
            anl = False, label = r'$\Delta x = 1 / %i$' % res)        
    
    pl.draw()
    del anl
    
mp.grid[1].legend(loc = 'lower right', frameon = False, ncol = 2)    
pl.savefig('test_convergence.png')
raw_input('')
    