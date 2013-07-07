"""

test_helium_chemistry.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Dec 26 18:37:48 2012

Description: Evolve helium species.

"""

import rt1d
import numpy as np
import pylab as pl
import chianti.core as cc

dims = 32
T = np.logspace(np.log10(5000), 6, dims)

# Initialize grid object
grid = rt1d.Grid(dims=dims)

# Set initial conditions
grid.set_physics(isothermal=True)
grid.set_chemistry(Z=[1,2], abundance=[1.0, 0.08])
grid.set_density(rho0=rt1d.Constants.m_H)
grid.set_temperature(T)
grid.set_ionization()#state='neutral')

# Initialize chemistry network / solver
chem = rt1d.Chemistry(grid, rt=False, dengo=False)

# Compute rate coefficients once (isothermal)
chem.chemnet.SourceIndependentCoefficients(grid.data['Tk'])

# Plot Equilibrium solution
np.seterr(all='ignore')
Teq = np.logspace(np.log10(np.min(T)), np.log10(np.max(T)), 500)
eqHe = cc.ioneq(2, Teq)
ax = pl.subplot(111)
ax.loglog(Teq, eqHe.Ioneq[0], color='k', ls='-')
ax.loglog(Teq, eqHe.Ioneq[1], color='k', ls='--')
ax.loglog(Teq, eqHe.Ioneq[2], color='k', ls=':')
ax.set_xlabel(r'$T \ (\mathrm{K})$')
ax.set_ylabel('Species Fraction')
ax.set_xlim(min(T), max(T))
ax.set_ylim(5e-9, 1.5)
pl.draw()

# Evolve chemistry
dt = rt1d.Constants.s_per_gyr
data = chem.Evolve(grid.data, t=0, dt=dt)

# Plot up solution
ax.scatter(T, data['he_1'], color='b', s=50, marker='o')                  
ax.scatter(T, data['he_2'], color='b', s=50, alpha=0.25, marker='o')     
ax.scatter(T, data['he_3'], color='b', s=50, facecolors='none', marker='o')            
pl.draw()    
raw_input('')

