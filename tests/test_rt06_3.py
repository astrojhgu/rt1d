"""

test_rt06_3.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Jan 18 17:00:27 2013

Description: 

"""

import rt1d
import matplotlib.pyplot as pl
from multiplot import multipanel

sim = rt1d.run.Simulation(problem_type=3, grid_cells=256,
    tables_dlogN=[0.05])
sim.run()

anl = rt1d.analyze.Simulation(sim.checkpoints)

# Plot solutions at 1, 3, 5, 15 Myr
t = [1, 3, 5, 15]
ls = [':', '--', '-.', '-']

mp = multipanel(dims=(2, 1))
    
# Plot initial conditions
mp.grid[0].semilogy(anl.grid.r_mid / anl.grid.length_units, 
    anl.data['dd0000']['h_1'], color='b', ls='-')
mp.grid[1].semilogy(anl.grid.r_mid / anl.grid.length_units, 
    anl.data['dd0000']['Tk'], color='b', ls='-')
    
ct = 0
for dd in anl.data.keys():
    t_code = anl.data[dd]['time'] / anl.pf['time_units']
    
    if t_code not in t: 
        continue
        
    this_t = int(t_code)

    mp.grid[0].semilogy(anl.grid.r_mid / anl.grid.length_units, anl.data[dd]['h_1'], 
        color='k', ls=ls[ct], 
        label = r'$t = %i \ \mathrm{Myr}$' % this_t)
    mp.grid[1].semilogy(anl.grid.r_mid / anl.grid.length_units, anl.data[dd]['Tk'], 
        color='k', ls=ls[ct])
    
    ct += 1

mp.grid[0].set_ylim(1e-3, 1.5)
mp.grid[1].set_ylim(10, 8e4)
                                        
for i in xrange(2):
    mp.grid[i].set_xlim(0.6, 1.0)
                            
mp.grid[0].set_xlabel(r'$x / L_{\mathrm{box}}$')    
mp.grid[0].set_ylabel('Neutral Fraction')
mp.grid[1].set_ylabel(r'Temperature $(K)$')    
mp.fix_ticks()

mp.grid[0].legend(loc='lower right', frameon=False)
                
pl.draw() 



