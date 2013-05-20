"""

test_continuous_radiation_3d.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Jan  3 15:10:46 2013

Description: Test 3-D lookup table construction and interpolation table 
generation.

"""

import rt1d
import pylab as pl
import numpy as np

grid = rt1d.Grid(length_units=3e23) # 100 kpc grid
    
# Set initial conditions
grid.set_physics(isothermal=0)
grid.set_chemistry(Z=[1,2], abundance=[1.0, 0.08])
grid.set_density(rho0=1e-27)

# Initialize radiation source
src_pars = {'problem_type': 12, 
    'spectrum_logNmin': [15]*3,
    'spectrum_logNmax': [20]*3,
    'spectrum_dlogN': [1.]*3}
src = rt1d.sources.RadiationSource(grid, **src_pars)

npts = 100
logN = [np.linspace(src.tab.logN[0].min(), src.tab.logN[0].max(), npts)]
logN.append(np.zeros(npts))
logN.append(np.zeros(npts))
logN = np.array(logN).T   
          
for i, table in enumerate(src.tables.keys()):
    # Sampled values
    pl.scatter(src.tab.logN[0], src.tabs[table][...,0,0], 
        color='k', s=50)
    pl.scatter(src.tab.logN[0], src.tabs[table][...,3,0], 
        color='b', s=50) 
        
    # Interpolated values    
    pl.plot(logN, src.tables[table](logN.copy()), color='k')
    
    # Axes limits
    pl.xlim(src.tab.logN[0].min() - 0.1, src.tab.logN[0].max() + 0.1)
    mi1, ma1 = src.tabs[table][...,0,0].min(), src.tabs[table][...,0,0].max()
    mi2, ma2 = src.tabs[table][...,3,0].min(), src.tabs[table][...,3,0].max()
    mi = min(mi1, mi2)
    ma = max(ma1, ma2)
    pl.ylim(mi-0.1, ma+0.1)
    
    # Axes labels
    pl.xlabel(r'Column Density $\log (N_{\mathrm{HI}} / \mathrm{cm}^{-2})$')
    pl.ylabel(table)
    pl.draw()
    
    raw_input('<enter> for next table')
    pl.close()
    