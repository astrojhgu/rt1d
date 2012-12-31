"""

test_hydrogen_chemistry.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Dec 26 18:37:48 2012

Description: 

"""

import rt1d
import numpy as np
import pylab as pl
import chianti.core as cc

dims = 32
T = np.logspace(3, 8, dims)

# Initialize grid object
grid = rt1d.Grid(dims = dims)

# Set initial conditions
grid.set_chem(Z = 8, abundance = 'sun_photospheric', isothermal = True)
grid.set_rho(rho0 = rt1d.Constants.m_H)
grid.set_T(T)
grid.set_x(state = 'neutral')  

# Initialize chemistry network / solver
chem = rt1d.Chemistry(grid, dengo = True)

# To compute timestep
timestep = rt1d.run.ComputeTimestep(grid)

# Plot Equilibrium solution
np.seterr(all = 'ignore')
Teq = np.logspace(np.log10(np.min(T)), np.log10(np.max(T)), 500)
eq = cc.ioneq(8, Teq)
ax = pl.subplot(111)
for i in xrange(eq.Ioneq.shape[0]):
    ax.loglog(Teq, eq.Ioneq[i], color = 'k', ls = '-')
ax.set_xlabel(r'$T \ (\mathrm{K})$')
ax.set_ylabel('Species Fraction')
ax.set_xlim(min(T), max(T))
ax.set_ylim(5e-9, 1.5)
pl.draw()

# Evolve chemistry
data = grid.data
dt = rt1d.Constants.s_per_myr / 1e3
dt_max = 2 * rt1d.Constants.s_per_myr
t = 0.0
tf = 1e2 * rt1d.Constants.s_per_myr

# Initialize progress bar
pb = rt1d.run.ProgressBar(tf)

while t <= tf:
    pb.update(t)
    data = chem.Evolve(data, dt = dt)  # kwargs not making it to RateEquations...
    t += dt 
    
    new_dt = timestep.IonLimited(chem.chemnet.q, chem.chemnet.dqdt)
    dt = min(min(min(new_dt, 2 * dt), dt_max), tf - t)

    if dt == 0:
        break

pb.finish()    
        
for ion in grid.all_ions:
    ax.scatter(T, data[ion], color = 'b', s = 50, 
        facecolors='none', marker = 'o')
pl.draw()    
raw_input('')



