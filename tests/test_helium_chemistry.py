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
T = np.logspace(np.log10(3e3), 6, dims)

# Initialize grid object
grid = rt1d.Grid(dims = dims)

# Set initial conditions
grid.set_chem(Z = [2], abundance = 'cosmic', isothermal = True)
grid.set_rho(rho0 = 1e-3 * rt1d.Constants.m_H)
grid.set_T(T)
grid.set_x(state = 'neutral')

# Initialize chemistry network / solver
chem = rt1d.Chemistry(grid, rt = False)

# Only need to calculate coefficients once for this test
chem.chemnet.SourceIndependentCoefficients(chem.grid.data['T'])

# To compute timestep
timestep = rt1d.run.ComputeTimestep(grid)

# Plot Equilibrium solution
np.seterr(all = 'ignore')
Teq = np.logspace(np.log10(np.min(T)), np.log10(np.max(T)), 500)
#eqH = cc.ioneq(1, Teq)
eqHe = cc.ioneq(2, Teq)
ax = pl.subplot(111)
#ax.loglog(Teq, eqH.Ioneq[0], color = 'k', ls = '-')
#ax.loglog(Teq, eqH.Ioneq[1], color = 'k', ls = '--')
ax.loglog(Teq, eqHe.Ioneq[0], color = 'k', ls = '-')
ax.loglog(Teq, eqHe.Ioneq[1], color = 'k', ls = '--')
ax.loglog(Teq, eqHe.Ioneq[2], color = 'k', ls = ':')
ax.set_xlabel(r'$T \ (\mathrm{K})$')
ax.set_ylabel('Species Fraction')
ax.set_xlim(min(T), max(T))
ax.set_ylim(5e-9, 1.5)
pl.draw()

# Evolve chemistry
data = grid.data
dt = rt1d.Constants.s_per_myr / 1e3
dt_max = 1e2 * rt1d.Constants.s_per_myr
t = 0.0
tf = 10 * rt1d.Constants.s_per_gyr

# Initialize progress bar
pb = rt1d.run.ProgressBar(tf)

while t <= tf:
    pb.update(t)
    data = chem.Evolve(data, dt = dt)
    t += dt 
    
    new_dt = timestep.IonLimited(chem.chemnet.q, chem.chemnet.dqdt)
    dt = min(min(min(new_dt, 2 * dt), dt_max), tf - t)

    if dt == 0:
        break

pb.finish()    

#ax.scatter(T, data['h_1'], color = 'b', s = 50, 
#    marker = 'o')
#ax.scatter(T, data['h_2'], color = 'b', s = 50, 
#    alpha = 0.5, marker = 'o')         
ax.scatter(T, data['he_1'], color = 'b', s = 50, 
    marker = 'o')
ax.scatter(T, data['he_2'], color = 'b', s = 50, 
    alpha = 0.5, marker = 'o') 
ax.scatter(T, data['he_3'], color = 'b', s = 50, 
    facecolors='none', marker = 'o')            
pl.draw()    
raw_input('')

