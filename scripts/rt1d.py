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

# If restart or batch do cool things
all_pfs = []
if sys.argv[1].strip('-') == 'r':
    IsRestart = True
    pf, ICs = rtm.ReadRestartFile(sys.argv[2])  
    all_pfs.append(pf)
    del pf
elif sys.argv[1].strip('-') == 'b':
    IsRestart = False
    f = open(sys.argv[2], 'r')
    for line in f: 
        pf = rtm.ReadParameterFile(line.strip())
        all_pfs.append(pf)
        del pf
    f.close()
else:
    IsRestart = False
    pf = rtm.ReadParameterFile(sys.argv[1])
    all_pfs.append(pf)
    del pf
    
# Print some output to the screen        
if rank == 0: 
    print "\nStarting rt1d..."
    print "Initializing {0} 1D radiative transfer calculation(s)...".format(len(all_pfs)) 
    if IsRestart: print "Restarting from {0}".format(sys.argv[2])
    print "Press ctrl-C to quit at any time.\n" 

start = time.time()

# Loop over parameter sets. 
for i, pf in enumerate(all_pfs):    
    TimeUnits = pf["TimeUnits"]
    StopTime = pf["StopTime"] * TimeUnits
    dt = pf["ODEMaxStep"] * TimeUnits
    dtDataDump = pf["dtDataDump"] * TimeUnits
            
    # Initialize grid and file system
    if IsRestart: data = ICs
    else:
        g = rtm.InitializeGrid(pf)   
        data = g.InitializeFields()
        
        if rank == 0:
            if pf["OutputDirectory"] != './':
                try: os.mkdir("{0}".format(pf["OutputDirectory"]))
                except OSError: 
                    os.system("rm -rf {0}".format(pf["OutputDirectory"]))
                    os.mkdir("{0}".format(pf["OutputDirectory"]))            
            made = True
        else: made = False
        
        if size > 1: MPI.COMM_WORLD.bcast(made, root = 0)    

    # Initialize integral tables
    iits = rtm.InitializeIntegralTables(pf, data)
    itabs = iits.TabulateRateIntegrals()
    if pf["ExitAfterIntegralTabulation"]: continue
                
    # Initialize radiation, write data, and monitor classes
    r = rtm.Radiate(pf, data, itabs, [iits.HIColumn, iits.HeIColumn, iits.HeIIColumn])
    w = rtm.WriteData(pf)
        
    # Figure out data dump times, write out initial dataset (or not if this is a restart).
    ddt = np.arange(0, StopTime + dtDataDump, dtDataDump)
    t = pf["CurrentTime"] * TimeUnits
    h = pf["ODEMaxStep"] * TimeUnits
    wct = int(t / dtDataDump) + 1
    if not IsRestart: w.WriteAllData(data, 0, t)
                    
    while t < StopTime:

        # Evolve photons
        data, h = r.EvolvePhotons(data, t, dt, h)
        t += dt
                
        # Write-out data, or don't                                        
        if t == ddt[wct]:
            if rank == 0: w.WriteAllData(data, wct, t)
            wct += 1
        elif (t + dt) > ddt[wct]:
            dt = ddt[wct] - t
        else:
            dt = dt
        
    elapsed = time.time() - start    
    print "Calculation {0} complete (output to {1}).  Elapsed time = {2} seconds.".format(i + 1, pf["OutputDirectory"], int(elapsed))





