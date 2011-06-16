"""
RT1D.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-10-14.

Description: Driver script for a 1D radiative transfer code.

"""

import sys, os, time
import numpy as np

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    print "Module mpi4py not found.  No worries, we'll just run in serial."
    rank = 0
    size = 1

# Class definitions
import rt1d.mods as rtm

all_pfs = {}

# Currently we only understand parameter files and the '-r' restart flag.  
if sys.argv[1] == '-r':
    IsRestart = True
    rf = "{0}/{1}".format(os.getcwd(), sys.argv[2])
    pf, ICs = rtm.ReadRestartFile(rf)
    all_pfs["{0}0000".format(pf["SavePrefix"])] = pf
    del pf

else:
    IsRestart = False
    pf = sys.argv[1]

    # Instantiate initialization class and build parameter space.
    ps = rtm.InitializeParameterSpace(pf)
    all_pfs = ps.AllParameterSets()
    del pf

if rank == 0: 
    print "\nStarting rt1d..."
    print "Initializing {0} 1D radiative transfer calculation(s)...".format(len(all_pfs)) 
    if IsRestart: print "Restarting from {0}".format(rf)
    print "Press ctrl-C to quit at any time.\n" 

# Loop over parameter sets. 
for i, pf in enumerate(all_pfs):
    if i % size != rank: continue

    start = time.time()
    
    this_pf = all_pfs[pf]
    this_pf["BaseName"] = pf
    TimeUnits = this_pf["TimeUnits"]
    StopTime = this_pf["StopTime"] * TimeUnits
    dt = this_pf["ODEMaxStep"] * TimeUnits
    dtDataDump = this_pf["dtDataDump"] * TimeUnits
        
    # Initialize grid and file system
    if IsRestart: data = ICs
    else:
        g = rtm.InitializeGrid(this_pf)   
        data = g.InitializeFields()
        
        try: os.mkdir("{0}".format(pf))
        except OSError: 
            os.system("rm -rf {0}".format(pf))
            os.mkdir("{0}".format(pf))

    # Initialize integral tables
    iits = rtm.InitializeIntegralTables(this_pf, data)
    itabs = iits.TabulateRateIntegrals()
    if this_pf["ExitAfterIntegralTabulation"]: continue
        
    # Initialize radiation, write data, and monitor classes
    r = rtm.Radiate(this_pf, data, itabs, [iits.HIColumn, iits.HeIColumn, iits.HeIIColumn])
    w = rtm.WriteData(this_pf)
    
    # Figure out data dump times, write out initial dataset (or not if this is a restart).
    ddt = np.arange(0, StopTime + dtDataDump, dtDataDump)
    t = this_pf["CurrentTime"] * TimeUnits
    h = this_pf["ODEMaxStep"] * TimeUnits
    wct = int(t / dtDataDump) + 1
    if not IsRestart: w.WriteAllData(data, 0, t)
                    
    while t < StopTime:

        # Evolve photons
        data, h = r.EvolvePhotons(data, t, dt, h)
        t += dt
                
        # Write-out data, or don't                                        
        if t == ddt[wct]:
            w.WriteAllData(data, wct, t)
            wct += 1
        elif (t + dt) > ddt[wct]:
            dt = ddt[wct] - t
        else:
            dt = dt
        
    elapsed = time.time() - start    
    print "Calculation {0} ({1}) complete.  Elapsed time = {2} seconds.".format(i + 1, pf, int(elapsed))





