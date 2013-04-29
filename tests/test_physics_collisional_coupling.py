"""

test_physics_collisional_coupling.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Nov  9 16:01:47 2012

Description: 

"""

   
import pylab as pl
import numpy as np
from rt1d.physics import Hydrogen

hydr = Hydrogen()

# Relevant temperature range
T = np.logspace(0, 4, 500)

# Plot coupling coefficients vs. temperature (tabulated and interpolated)
pl.scatter(hydr.tabulated_coeff['T_H'], 
    hydr.tabulated_coeff['kappa_H'], color = 'k')
pl.scatter(hydr.tabulated_coeff['T_e'], 
    hydr.tabulated_coeff['kappa_e'], color = 'k')
pl.loglog(T, hydr.kappa_H(T), color = 'k', ls = '-', 
    label = r'$\kappa_{10}^{\mathrm{HH}}$')
pl.loglog(T, hydr.kappa_e(T), color = 'k', ls = '--', 
    label = r'$\kappa_{10}^{\mathrm{eH}}$') 
pl.xlim(1, 1e4)
pl.ylim(1e-13, 2e-8)
pl.xlabel(r'$T \ (\mathrm{K})$')
pl.ylabel(r'$\kappa \ (\mathrm{cm}^3 \mathrm{s}^{-1})$')
pl.legend(loc = 'lower right', frameon = False)
pl.annotate('Points from Zygelman (2005)', (1.5, 1e-8), ha = 'left')
pl.annotate('Lines are spline fit to points', (1.5, 5e-9), ha = 'left')
raw_input('')
pl.close()     
   