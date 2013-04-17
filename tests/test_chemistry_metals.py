"""

test_chemistry_metals.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Dec 26 18:37:48 2012

Description: Evolve metal species to equilibrium using dengo-based solver.

"""

import rt1d
import numpy as np
import pylab as pl
import chianti.core as cc

#
##
Z = 8
dims = 32
##
#

colors = ['k', 'b', 'g', 'r', 'c', 'm', 'y'] * 2

T = np.logspace(4, 8, dims)

# Initialize grid object
grid = rt1d.Grid(dims=dims)

# Set initial conditions - one particle per cc
grid.set_physics(isothermal=True)
grid.set_chemistry(Z=Z)
grid.set_density(rho0=10*rt1d.Constants.m_H*2*Z)
grid.set_temperature(T)
grid.set_ionization()

# Initialize chemistry network / solver
chem = rt1d.Chemistry(grid, dengo=True)

# Plot equilibrium solution
np.seterr(all = 'ignore')
Teq = np.logspace(np.log10(np.min(T)), np.log10(np.max(T)), 128)
eq = cc.ioneq(Z, Teq)
ax = pl.subplot(111)
for i in xrange(eq.Ioneq.shape[0]):
    ax.loglog(Teq, eq.Ioneq[i], color=colors[i], ls = '-')
ax.set_xlabel(r'$T \ (\mathrm{K})$')
ax.set_ylabel('Species Fraction')
ax.set_xlim(min(T), max(T))
ax.set_ylim(5e-9, 1.5)
pl.draw()

# Evolve chemistry in one big step
dt = rt1d.Constants.s_per_gyr
data = chem.Evolve(grid.data, t=0, dt=dt)
    
# Plot up solution
for i, ion in enumerate(grid.all_ions):
    ax.scatter(T, data[ion], color=colors[i], s=50, 
        facecolors='none', marker='o')
pl.draw()   
raw_input('')



