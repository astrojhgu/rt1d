"""
rt1d.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-10-14.

Description: Driver script for a 1D radiative transfer code.

"""

import sys
import numpy as np

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except:
    ImportError("Module mpi4py not found.  No worries, we'll just run in serial.")
    rank = 0
    size = 1

# Class definitions
from InitializeParameterSpace import *
from InitializeGrid import *
from Radiate import *
from WriteData import *

# Retrieve name of parameter file from command line.
pf = sys.argv[1]

# Instantiate initialization class and build parameter space.
ps = InitializeParameterSpace(pf)
all_pfs = ps.AllParameterSets()
del pf

if rank == 0: 
    print "\nStarting rt1d..."
    print "Initializing {0} 1D radiative transfer calculation(s)...".format(len(all_pfs)) 
    print "Press ctrl-C to quit at any time.\n" 

# Loop over parameter sets. 
for i, pf in enumerate(all_pfs):
    if i % size != rank: continue
    this_pf = all_pfs[pf]
    
    # Instantiate necessary classes.
    g = InitializeGrid(this_pf)   
    r = Radiate(this_pf)
    w = WriteData(this_pf)
    data = g.InitializeFields()
    
    # Figure out data dump times, write out initial dataset.
    dd = np.arange(0, this_pf["StopTime"] + this_pf["dtDataDump"], this_pf["dtDataDump"])
    w.WriteAllData(data, 0.0)    
    
    t = 0.0
    dt = this_pf["InitialTimestep"]
    wct = 1
    while t < this_pf["StopTime"]:
        data, dt = r.EvolvePhotons(data, t, dt)
        
        if t == dd[wct]:
            w.WriteAllData(data, t)
            wct += 1
        elif (t + dt) > dd[wct]:
            dt = dd[wct] - t
        
        t += dt
        
    del g, r, w, data
    print "Calculation {0} complete.".format(i + 1)

    


