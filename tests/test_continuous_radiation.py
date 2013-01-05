"""

test_continuous_radiation.py

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
grid.set_chem(Z = [1], abundance = 'cosmic', 
    isothermal = 0)
grid.set_rho(rho0 = 1e-27)

# Initialize radiation source
src_pars = {'source_type': 1, 'isothermal': 0, 'problem_type': 2}
src = rt1d.sources.RadiationSourceIdealized(grid, **src_pars)

# Initialize table
src.create_integral_table(dims = 10)

# Plot Phi
pl.scatter(10**src.tab.N[0], 10**src.tabs['logPhi_h_1'], 
    color = 'k', s = 50)

# Plot interpolated result
N = np.linspace(min(src.tab.N[0]), max(src.tab.N[0]), 100)
pl.loglog(10**N, 10**src.tables['logPhi_h_1'](N), color = 'k')

pl.xlabel(r'Column Density $N_{\mathrm{HI}} \ (\mathrm{cm}^{-2})$')
pl.ylabel(r'$\Phi_{\mathrm{HI}}$')

raw_input('')
pl.close()

pl.scatter(10**src.tab.N[0], 10**src.tabs['logPsi_h_1'], 
    color = 'k', s = 50)
pl.loglog(10**N, 10**src.tables['logPsi_h_1'](N), color = 'k')
pl.xlabel(r'Column Density $N_{\mathrm{HI}} \ (\mathrm{cm}^{-2})$')
pl.ylabel(r'$\Psi_{\mathrm{HI}}$')

raw_input('')
pl.close()

pl.scatter(10**src.tab.N[0], 10**src.tabs['logTau'],
    color = 'k', s = 50)
pl.loglog(10**N, 10**src.tables['logTau'](N), color = 'k')
pl.xlabel(r'Column Density $N_{\mathrm{HI}} \ (\mathrm{cm}^{-2})$')
pl.ylabel(r'$\tau$')

raw_input('')





