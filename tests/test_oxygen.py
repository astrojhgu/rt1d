"""

test_inits.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Sep 20 13:18:18 2012

Description: Initialize neutral oxygen-only medium.  Evolve to equilibrium.

"""

import pylab as pl
import numpy as np
import h5py, chianti, rt1d

try:
    from progressbar import *
    gotpb = True
except ImportError:
    gotpb = False

dims = 32
TimeUnits = rt1d.Constants.s_per_myr
OutputFile = 'oxygen_test.h5'

# Initialize grid object
grid = rt1d.Grid(dims = dims)

# Set initial conditions
grid.set_chem(Z = [8], abundance = 'cosmic', isothermal = True)
grid.set_rho(rho0 = rt1d.Constants.m_H)
grid.set_T(np.logspace(4, np.log10(5e6), dims))
grid.set_x(state = 'neutral')  

# Initialize chemistry network / solver
chem = rt1d.Chemistry(grid)
 
# Open data dump file and write-out initial conditions    
f = h5py.File(OutputFile, 'a')    
f.attrs.create('TimeStep', TimeUnits / 1e4)

grp = f.create_group('dd0000')
for key in grid.data.keys():
    grp.create_dataset(key, data = grid.data[key])
    
# Set up progressbar
if gotpb: # This could take a while
    widget = ["oxygen:", Percentage(), ' ', Bar(marker = RotatingMarker()), ' ', ETA(), ' ']
    pbar = ProgressBar(widgets = widget, maxval = float(1000)).start()                        

# Evolve chemistry for 10^5 years
new_data = grid.data.copy()
for i in np.arange(1, 1000):
    
    # Solve - dt = 100 years
    new_data = chem.Evolve(new_data, TimeUnits / 1e4)

    # Write out data
    f = h5py.File(OutputFile, 'a')    
    f.attrs.create('TimeStep', TimeUnits / 1e4)
    grp = f.create_group('dd%s' % str(i).zfill(4))
    for key in new_data.keys():
        grp.create_dataset(key, data = new_data[key])
    
    f.close()    
    
    if gotpb:
        pbar.update(float(i))
        
if gotpb:
    pbar.finish()
