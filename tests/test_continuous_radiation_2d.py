"""

test_continuous_radiation_2d.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Jan  3 15:10:46 2013

Description: 

"""

import rt1d
import pylab as pl
import numpy as np

grid = rt1d.Grid(length_units = 3e23) # 100 kpc grid
    
# Set initial conditions
grid.set_physics(isothermal=0)
grid.set_chemistry(Z=1)
grid.set_density(rho0 = 1e-27)

# Initialize radiation source
src_pars = {'problem_type': 2, 'photon_conserving': 1, 
    'secondary_ionization': 2}
src = rt1d.sources.RadiationSource(grid, **src_pars)

logN = np.array([np.linspace(min(src.tab.logN[0]), max(src.tab.logN[0]), 100)]).T
logx1 = np.ones(100) * src.tab.logx[0]
logx2 = np.ones(100) * src.tab.logx[-1]

tables = ['logPhiHat_h_1', 'logPsiHat_h_1',     
          'logPhiWiggle_h_1_h_1', 'logPsiWiggle_h_1_h_1']

labels = [r'$\hat{\Phi}_{\mathrm{HI}}$',
          r'$\hat{\Psi}_{\mathrm{HI}}$',
          r'$\tilde{\Phi}_{\mathrm{HI}}$',
          r'$\tilde{\Psi}_{\mathrm{HI}}$']
          
for i, table in enumerate(tables):
    pl.scatter(src.tab.N[0], 10**src.tabs[table][..., 0], 
        color = 'k', s = 50)
    pl.scatter(src.tab.N[0], 10**src.tabs[table][..., -1], 
        color = 'b', s = 50)    
    pl.loglog(10**logN, 10**src.tables[table](logN, logx1), color = 'k')
    pl.loglog(10**logN, 10**src.tables[table](logN, logx2), color = 'b')
    pl.xlabel(r'Column Density $N_{\mathrm{HI}} \ (\mathrm{cm}^{-2})$')
    pl.ylabel(labels[i])
    pl.draw()
    
    raw_input('<enter> for %s.' % table)
    pl.close()
    