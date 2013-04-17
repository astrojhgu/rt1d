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
from multiplot import multipanel

#
##
Z = [6,8]
dims = 32
abundance = [1., 0.0004]  # roughly sun_photospheric ratio for carbon/oxygen
##
#

colors = ['k', 'b', 'g', 'r', 'c', 'm', 'y'] * 2

T = np.logspace(4, 8, dims)

# Initialize grid object
grid = rt1d.Grid(dims = dims)

# Set initial conditions - densities corresponding to 1 H-atom per cc
grid.set_physics(isothermal=True)
grid.set_chemistry(Z=Z, abundance=abundance)
grid.set_density(rho0=10*rt1d.Constants.m_H * 12)
grid.set_temperature(T)
grid.set_ionization()

# Initialize chemistry network / solver
chem = rt1d.Chemistry(grid, dengo=True)

# Plot equilibrium solution (nicely)
np.seterr(all = 'ignore')
Teq = np.logspace(np.log10(np.min(T)), np.log10(np.max(T)), 128)

mp = multipanel(dims=(len(grid.Z),1), panel_size=(1, len(grid.Z)), share_all=False)
for i, element in enumerate(grid.elements):
    eq = cc.ioneq(grid.Z[i], Teq)
    
    for j in xrange(eq.Ioneq.shape[0]):
        mp.grid[i].loglog(Teq, eq.Ioneq[j], color=colors[j])
    
    del eq    

for i in xrange(len(grid.Z)):
    mp.grid[i].set_ylabel(r'Species Fractions for $Z=%i$' % grid.Z[i])
    mp.grid[i].set_xlim(min(T), max(T))
    mp.grid[i].set_ylim(5e-9, 1.5)
    
mp.global_xlabel(r'$T \ (\mathrm{K})$')
mp.fix_ticks()

# Evolve chemistry
dt = rt1d.Constants.s_per_gyr
data = chem.Evolve(grid.data, t=0, dt=dt)

# Plot solutions
for i, element in enumerate(grid.elements):    
    for j, ion in enumerate(grid.ions_by_parent[element]):
        mp.grid[i].scatter(T, data[ion], color=colors[j], s=50,
            facecolors='none', marker = 'o')
    
pl.draw()
raw_input('')



