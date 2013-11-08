"""

generate_lookup_table.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sat May 25 16:15:57 2013

Description: Tabulate integral quantities in N-D, where N=len(Z).

"""

import rt1d
import pylab as pl
import numpy as np

# 
fn = 'bb_1e5_h_he.hdf5'
#

# Initialize radiation source
src_pars = \
    {
    'problem_type': 12, 
    'tables_logNmin': [15]*3,
    'tables_logNmax': [20]*3,
    'tables_dlogN': [0.1]*3
    }

grid = rt1d.static.Grid(length_units=3e23) # 100 kpc grid
    
# Set initial conditions
grid.set_physics(isothermal=0)
grid.set_chemistry(Z=[1,2], abundance=[1.0, 0.08])
grid.set_density(rho0=1e-27)

# Initialize radiation source - create lookup table
src = rt1d.sources.RadiationSource(grid, **src_pars)

src.tab.save(fn)