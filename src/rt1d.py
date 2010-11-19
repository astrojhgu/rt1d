#! /usr/bin/env python
"""
rt1d.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-10-14.

Description: Driver script for a 1D radiative transfer code.

"""

import sys, os, time
import numpy as np
from progressbar import *

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
from ReadRestartFile import *
from MonitorSimulation import *

all_pfs = {}

# Currently we only understand parameter files and the '-r' restart flag.  
if sys.argv[1] == '-r':
    IsRestart = True
    rf = "{0}/{1}".format(os.getcwd(), sys.argv[2])
    pf, ICs = ReadRestartFile(rf)
    all_pfs["{0}0000".format(pf["SavePrefix"])] = pf
    del pf

else:
    IsRestart = False
    pf = sys.argv[1]

    # Instantiate initialization class and build parameter space.
    ps = InitializeParameterSpace(pf)
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
    dt = this_pf["InitialTimestep"] * TimeUnits
    dtDataDump = this_pf["dtDataDump"] * TimeUnits
    
    # Widget for progressbar.
    widget = ["rt1d: ", Percentage(), ' ', Bar(marker = RotatingMarker()), ' ', ETA(), ' ']
            
    # Initialize grid and file system
    if IsRestart: data = ICs
    else:
        g = InitializeGrid(this_pf)   
        data = g.InitializeFields()
        
        try: os.mkdir("{0}".format(pf))
        except OSError: 
            os.system("rm -rf {0}".format(pf))
            os.mkdir("{0}".format(pf))

    # Initialize integral tables
    iits = InitializeIntegralTables(this_pf, data)
    itabs = iits.TabulateRateIntegrals()
    
    # Initialize radiation, write data, and monitor classes
    r = Radiate(this_pf, itabs, [iits.HIColumn, iits.HeIColumn, iits.HeIIColumn])
    w = WriteData(this_pf)
    ms = MonitorSimulation(this_pf)
    
    # Figure out data dump times, write out initial dataset (or not if this is a restart).
    ddt = np.arange(0, StopTime + dtDataDump, dtDataDump)
    t = this_pf["CurrentTime"] * TimeUnits
    wct = int(t / dtDataDump) + 1
    if not IsRestart: w.WriteAllData(data, 0, t)
                
    while t <= StopTime:

        if t == 0 and rank == 0: print "rt1d:  t = 0.0 / {0}".format(StopTime / TimeUnits)

        # Progress bar
        if rank == 0:
            pbar = ProgressBar(widgets = widget, maxval = dtDataDump / TimeUnits).start()
            try: pbar.update((t / TimeUnits) - ((wct - 1) * (dtDataDump / TimeUnits)))
            except AssertionError: pass        
        
        # Evolve photons
        data, dt = r.EvolvePhotons(data, t, dt)
        if rank == 0 and this_pf["MonitorSimulation"]: ms.Monitor(data, t / TimeUnits)
               
        # Write-out data, or don't                                        
        if t == ddt[wct]:
            if rank == 0: pbar.finish()
            w.WriteAllData(data, wct, t)
            if rank == 0: print "rt1d:  t = {0} / {1}".format(t / TimeUnits, StopTime / TimeUnits)
            wct += 1
        elif (t + dt) > ddt[wct]:
            dt = ddt[wct] - t
        else:
            dt = dt

        t += dt
        
    elapsed = time.time() - start    
    print "Calculation {0} ({1}) complete.  Elapsed time = {2} seconds.".format(i + 1, pf, int(elapsed))

    


