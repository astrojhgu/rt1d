"""

test_continuous_radiation_1d.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Jan  3 15:10:46 2013

Description: 

"""

import rt1d
import pylab as pl
import numpy as np

grid = rt1d.static.Grid(length_units=3e23) # 100 kpc grid
    
# Set initial conditions
grid.set_physics(isothermal=0)
grid.set_chemistry(Z=1)
grid.set_density(rho0=1e-27)

# Initialize radiation source
src_pars = {'problem_type': 2, 'photon_conserving': 1}
src = rt1d.sources.RadiationSource(grid=grid, **src_pars)

# Plot Phi
pl.scatter(src.tab.N[0], 10**src.tabs['logPhi_h_1'], 
    color = 'k', s = 50)

# Plot interpolated result
logN = np.array([np.linspace(min(src.tab.logN[0]), max(src.tab.logN[0]), 100)]).T
pl.loglog(10**logN, 10**src.tables['logPhi_h_1'](logN), color = 'k')

pl.xlabel(r'Column Density $N_{\mathrm{HI}} \ (\mathrm{cm}^{-2})$')
pl.ylabel(r'$\Phi_{\mathrm{HI}}$')
pl.draw()

raw_input('<enter> for Psi.')
pl.close()

# Plot Psi
pl.scatter(src.tab.N[0], 10**src.tabs['logPsi_h_1'], color = 'k', s = 50)
pl.loglog(10**logN, 10**src.tables['logPsi_h_1'](logN), color = 'k')
pl.xlabel(r'Column Density $N_{\mathrm{HI}} \ (\mathrm{cm}^{-2})$')
pl.ylabel(r'$\Psi_{\mathrm{HI}}$')
pl.draw()

raw_input('<enter> for optical depth.')
pl.close()

# Plot optical depth
pl.scatter(src.tab.N[0], 10**src.tabs['logTau'], color = 'k', s = 50)
pl.loglog(10**logN, 10**src.tables['logTau'](logN), color = 'k')
pl.xlabel(r'Column Density $N_{\mathrm{HI}} \ (\mathrm{cm}^{-2})$')
pl.ylabel(r'$\tau$')
pl.draw()





