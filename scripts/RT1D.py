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
        if not line.strip(): continue
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
    
    if (i % size != rank) and (pf["ParallelizationMethod"] == 2): continue
     
    TimeUnits = pf["TimeUnits"]
    LengthUnits = pf["LengthUnits"]
    GridDimensions = pf["GridDimensions"]
    StopTime = pf["StopTime"] * TimeUnits
    dtDataDump = pf["dtDataDump"] * TimeUnits
    StartCell = int(pf["StartRadius"] * GridDimensions)
    r_0 = LengthUnits * StartCell / GridDimensions
    
    # Print out cell-crossing and box-crossing times for convenience
    if rank == 0:
        print "Cell-crossing time = {0} (years), {1} (code)".format(LengthUnits / GridDimensions / 29979245800.0 / 31557600.0, \
            LengthUnits / GridDimensions / 29979245800.0 / TimeUnits)
        print "Box-crossing time = {0} (years), {1} (code)\n".format(LengthUnits / 29979245800.0 / 31557600.0, LengthUnits / 29979245800.0 / TimeUnits)
                        
    # Initialize grid and file system
    if IsRestart: data = ICs
    else:
        g = rtm.InitializeGrid(pf)   
        data = g.InitializeFields()
                
        if i % size == rank:
            if pf["OutputDirectory"] != './':
                try: os.mkdir("{0}".format(pf["OutputDirectory"]))
                except OSError: pass        
            made = True
        else: made = False
        
        # Wait here if parallelizing over grid
        if size > 1 and pf["ParallelizationMethod"] == 1: MPI.COMM_WORLD.bcast(made, root = 0)    

    # Initialize integral tables
    iits = rtm.InitializeIntegralTables(pf, data)
    itabs = iits.TabulateRateIntegrals()        
    if pf["ExitAfterIntegralTabulation"]: continue
                
    # Initialize radiation and write data classes
    r = rtm.Radiate(pf, data, itabs, [iits.HIColumn, iits.HeIColumn, iits.HeIIColumn])
    w = rtm.WriteData(pf)
    
    # Compute timestep
    if IsRestart or pf["HIIRestrictedTimestep"] == 0: dt = pf["CurrentTimestep"] * TimeUnits
    else:
        
        if pf["MultiSpecies"]:
            indices = r.Interpolate.GetIndices3D(3 * [1e-10])
        else:
            indices = None    
            
        Gamma = r.IonizationRateCoefficientHI(3 * [1e-10], 0, data['HIDensity'][StartCell], data['HeIDensity'][StartCell], data["ElectronDensity"][StartCell], \
            data['Temperature'][StartCell], r_0, r.rs.BolometricLuminosity(0), indices)        
        alpha = 2.6e-13 * (data['Temperature'][StartCell] / 1.e4)**-0.85  
                
        # Shapiro et al. 2004 - override initial timestep in parameter file
        dt = pf["MaxHIIChange"] * data['HIDensity'][StartCell] / \
            np.abs(data['HIDensity'][StartCell] * Gamma - data["ElectronDensity"][StartCell]**2 * alpha)
    
        # Make it even smaller just to play it safe if helium is involved
        if pf["MultiSpecies"]: dt *= 0.1   
            
    dt = min(dt, StopTime)
            
    # Figure out data dump times, write out initial dataset (or not if this is a restart).
    ddt = np.arange(0, StopTime + dtDataDump, dtDataDump)
    t = pf["CurrentTime"] * TimeUnits
    h = dt
    wct = int(t / dtDataDump) + 1
    if not IsRestart: 
        if rank == 0: 
            w.WriteAllData(data, 0, t, dt)
            
    if size > 1 and pf["ParallelizationMethod"] == 1: MPI.COMM_WORLD.bcast(wct, root = 0)         
    
    # If we want to store timestep evolution, setup dump file
    if pf["OutputTimestep"]: 
        if IsRestart: fdt = open('{0}/timestep_evolution.dat'.format(pf["OutputDirectory"]), 'a')
        else: fdt = open('{0}/timestep_evolution.dat'.format(pf["OutputDirectory"]), 'w')
        print >> fdt, '# t  dt'
        fdt.close()
    
    if size > 1: 
        lb = list(np.linspace(pf["GridDimensions"] / size, pf["GridDimensions"], size))
        lb.insert(0, StartCell)
    else: 
        lb = None
    
    # Solve radiative transfer                
    while t < StopTime:
                                
        if pf["OutputTimestep"]: 
            fdt = open('{0}/timestep_evolution.dat'.format(pf["OutputDirectory"]), 'a')
            print >> fdt, t / TimeUnits, dt / TimeUnits
                                
        # Ensure we land on our data dump times exactly
        if (t + dt) > ddt[wct]: 
            dt = ddt[wct] - t
            tnow = t + dt
            write_now = True
        elif (t + dt) == ddt[wct]: 
            tnow = t + dt
            write_now = True
        else: write_now = False

        # Evolve photons
        data, h, newdt, lb = r.EvolvePhotons(data, t, dt, min(h, dt), lb)
        t += dt
        dt = newdt # dt for the next timestep
                               
        # Write-out data                                        
        if write_now:
            wrote = False
            if i == rank: 
                w.WriteAllData(data, wct, tnow, dt)
                wrote = True
            wct += 1
           
        # Don't move on until root processor has written out data    
        if size > 1 and pf["ParallelizationMethod"] == 1: MPI.COMM_WORLD.bcast(write_now, root = 0)    
    
    if pf["OutputTimestep"]: 
        print >> fdt, ""
        fdt.close()
                
    elapsed = time.time() - start    
    print "Calculation {0} complete (output to {1}).  Elapsed time = {2} hours.".format(i + 1, pf["OutputDirectory"], round(elapsed / 3600., 2))

    f = open('{0}/RunComplete'.format(pf["OutputDirectory"]), 'w')
    print >> f, '# walltime (s)'
    print >> f, elapsed
    print >> f, ''
    f.close()

print "Successful run. Exiting."