"""

test_hydrogen_chemistry.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Dec 26 18:37:48 2012

Description: 

"""

import rt1d
import numpy as np
import matplotlib.pyplot as pl

src = 'fk94'
dims = 64
T = np.logspace(3, 5, dims)

# Initialize grid object
grid = rt1d.static.Grid(dims=dims)

# Set initial conditions
grid.set_physics(isothermal=True)
grid.set_chemistry(Z=1)
grid.set_density(rho0=rt1d.physics.Constants.m_H)
grid.set_ionization(state='neutral')  
grid.set_temperature(T)

# Initialize chemistry network / solver
chem = rt1d.evolve.Chemistry(grid, rt=False, dengo=False, rate_src=src)

# Compute rate coefficients once (isothermal)
chem.chemnet.SourceIndependentCoefficients(grid.data['Tk'])

# To compute timestep
timestep = rt1d.run.ComputeTimestep(grid)

# Evolve chemistry
data = grid.data
dt = rt1d.physics.Constants.s_per_myr / 1e3
dt_max = 1e2 * rt1d.physics.Constants.s_per_myr
t = 0.0
tf = rt1d.physics.Constants.s_per_gyr

# Initialize progress bar
pb = rt1d.util.ProgressBar(tf)
pb.start()

while t <= tf:
    pb.update(t)
    data = chem.Evolve(data, t=t, dt=dt)
    t += dt 
    
    new_dt = timestep.Limit(chem.chemnet.q, chem.chemnet.dqdt)
    dt = min(min(min(new_dt, 2 * dt), dt_max), tf - t)

    if dt == 0:
        break

pb.finish()    
                
ax = pl.subplot(111)        
ax.loglog(T, data['h_1'], color='k', ls='-')
ax.loglog(T, data['h_2'], color='k', ls='--')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$T \ (\mathrm{K})$')
ax.set_ylabel('Species Fraction')
ax.set_ylim(1e-4, 2)
pl.draw()    




