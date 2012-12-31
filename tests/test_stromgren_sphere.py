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

dims = 64

# Initialize grid object
grid = rt1d.Grid(dims = dims, 
    **{'LengthUnits': 6.6 * rt1d.Constants.cm_per_kpc})

# Set initial conditions
grid.set_chem(isothermal = True)
grid.set_rho(rho0 = 1e-3 * rt1d.Constants.m_H)
grid.set_T(1e4)
grid.set_x(Z = 1, x = 1.2e-3)  

# Initialize radiation source and radiative transfer solver
rs = rt1d.sources.RadiationSourceIdealized(**{'source_type': 0,
    'spectrum_type': 0, 'spectrum_qdot': 5e48, 'spectrum_E': [13.60001]})
rt = rt1d.Radiation(grid, rs)

# To compute timestep
timestep = rt1d.run.ComputeTimestep(grid)

# Evolve chemistry + radiation
data = grid.data
dt = rt1d.Constants.s_per_myr / 1e6
dt_max = 10 * rt1d.Constants.s_per_myr
t = 0.0
tf = 5e2 * rt1d.Constants.s_per_myr

pb = rt1d.run.ProgressBar(tf)

# Only need to calculate coefficients once for this test
nothing = rt.chem.chemnet.SourceIndependentCoefficients(rt.chem.grid.data['T'])

while t <= tf:

    data = rt.Evolve(data, t = t, dt = dt)
    pb.update(t)
    
    t += dt 
    
    new_dt = timestep.IonLimited(rt.chem.chemnet.q, rt.chem.chemnet.dqdt)
    dt = min(min(min(new_dt, 2 * dt), dt_max), tf - t)

    if dt == 0:
        break

pb.finish()    
        
pl.semilogy(grid.r_mid / grid.pf['LengthUnits'], data['h_1'], color = 'k', ls = '-')
pl.semilogy(grid.r_mid / grid.pf['LengthUnits'], data['h_2'], color = 'k', ls = '--')
pl.ylim(1e-3, 1.5)
raw_input('')



