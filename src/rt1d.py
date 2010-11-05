"""
rt1d.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-10-14.

Description: Driver script for a 1D radiative transfer code.

"""

import sys, os
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
from InitializeIntegralTables import *
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
    
    TimeUnits = this_pf["TimeUnits"]
    StopTime = this_pf["StopTime"] * TimeUnits
    dt = this_pf["InitialTimestep"] * TimeUnits
    dtDataDump = this_pf["dtDataDump"] * TimeUnits
        
    try: os.mkdir("{0}".format(pf))
    except OSError: 
        os.system("rm -rf {0}".format(pf))
        os.mkdir("{0}".format(pf))
    
    # Initialize grid
    g = InitializeGrid(this_pf)   
    data = g.InitializeFields()

    # Initialize integral tables
    iits = InitializeIntegralTables(this_pf, data)
    itabs = iits.TabulateRateIntegrals()
    
    # Initialize radiation and write data classes
    r = Radiate(this_pf, itabs, [iits.HIColumn, iits.HeIColumn, iits.HeIIColumn])
    w = WriteData(this_pf)
    
    # Figure out data dump times, write out initial dataset.
    ddt = np.arange(0, StopTime + dtDataDump, dtDataDump)
    w.WriteAllData(data, pf, 0, 0)
        
    t = 0.0
    wct = 1
    while t <= StopTime:
        data, dt = r.EvolvePhotons(data, t, dt)
                        
        print "t = {0}".format(t / TimeUnits)                
                        
        if t == ddt[wct]:
            w.WriteAllData(data, pf, wct, t)
            wct += 1
        elif (t + dt) > ddt[wct]:
            dt = ddt[wct] - t
        else:
            dt = dt

        t += dt
        
    del g, r, w, data
    print "Calculation {0} ({1}) complete.".format(i + 1, pf)

    


