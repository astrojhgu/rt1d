"""

test_continuous_radiation_1d.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Jan  3 15:10:46 2013

Description: 

"""

import rt1d
import matplotlib.pyplot as pl
import numpy as np

grid = rt1d.static.Grid(length_units=3e23) # 100 kpc grid
    
# Set initial conditions
grid.set_physics(isothermal=0)
grid.set_chemistry(Z=1)
grid.set_density(rho0=1e-27)

# Initialize radiation source
src_pars = {'problem_type': 2, 'photon_conserving': 1}
src = rt1d.sources.RadiationSource(grid=grid, **src_pars)

fig1 = pl.figure(1); ax1 = fig1.add_subplot(111)
fig2 = pl.figure(2); ax2 = fig2.add_subplot(111)
fig3 = pl.figure(3); ax3 = fig3.add_subplot(111)

# Plot Phi
ax1.scatter(src.tab.N[0], 10**src.tabs['logPhi_h_1'], color='k', s=50,
    facecolors='none')

# Plot interpolated result
logN = np.array([np.linspace(min(src.tab.logN[0]), max(src.tab.logN[0]), 100)]).T
ax1.loglog(10**logN, 10**src.tables['logPhi_h_1'](logN), color = 'k')

ax1.set_xlabel(r'Column Density $N_{\mathrm{HI}} \ (\mathrm{cm}^{-2})$')
ax1.set_ylabel(r'$\Phi_{\mathrm{HI}}$')

# Plot Psi
ax2.scatter(src.tab.N[0], 10**src.tabs['logPsi_h_1'], color='k', s=50,
    facecolors='none')
ax2.loglog(10**logN, 10**src.tables['logPsi_h_1'](logN), color='k')
ax2.set_xlabel(r'Column Density $N_{\mathrm{HI}} \ (\mathrm{cm}^{-2})$')
ax2.set_ylabel(r'$\Psi_{\mathrm{HI}}$')

# Plot optical depth
ax3.scatter(src.tab.N[0], 10**src.tabs['logTau'], color='k', s=50,
    facecolors='none')
ax3.loglog(10**logN, 10**src.tables['logTau'](logN), color='k')
ax3.set_xlabel(r'Column Density $N_{\mathrm{HI}} \ (\mathrm{cm}^{-2})$')
ax3.set_ylabel(r'$\tau$')
pl.draw()





