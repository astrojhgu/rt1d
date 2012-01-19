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
    
def Shine(pf, r = None, IsRestart = False):
    """
    w00t.
    """  
    
    # Convert parameter file to list of parameter files
    if type(pf) is not list:
        all_pfs = [pf]  
    else:
        all_pfs = pf    
        
    # Print some output to the screen        
    if rank == 0: 
        print "\nStarting rt1d..."
        print "Initializing {0} 1D radiative transfer calculation(s)...".format(len(all_pfs)) 
        if IsRestart: 
            print "Restarting from {0}".format(sys.argv[2])
        print "Press ctrl-C to quit at any time.\n" 
    
    # Start clock
    start = time.time()
    
    # Loop over parameter sets (may just be one, but what the heck). 
    for i, pf in enumerate(all_pfs):  
        
        # Only one parameter file per processor if ParallelizationMethod = 2.    
        if pf["ParallelizationMethod"] == 2:
            if i % size != rank: continue
            
        # Check for parameter conflicts    
        conflicts, errmsg = rtm.CheckForParameterConflicts(pf)
        if conflicts:
            print 'ERROR -- PARAMETER VALUES IN CONFLICT:'
            for msg in errmsg:
                print msg
            print '\n'    
            raise Exception()   
              
        # Read in a few things for convenience, convert times to real units     
        TimeUnits = pf["TimeUnits"]
        LengthUnits = pf["LengthUnits"]
        StartRadius = pf["StartRadius"]
        GridDimensions = pf["GridDimensions"]
        StopTime = pf["StopTime"] * TimeUnits
        dtDataDump = pf["dtDataDump"] * TimeUnits
        
        # Print out cell-crossing and box-crossing times for convenience
        if rank == 0:
            print "Cell-crossing time = {0} (years), {1} (code)".format(LengthUnits / GridDimensions / \
                29979245800.0 / 31557600.0, LengthUnits / GridDimensions / 29979245800.0 / TimeUnits)
            print "Box-crossing time = {0} (years), {1} (code)\n".format(LengthUnits / 29979245800.0 / \
                31557600.0, LengthUnits / 29979245800.0 / TimeUnits)
                            
        # Initialize grid and file system
        if IsRestart: 
            data = ICs
        else:
            g = rtm.InitializeGrid(pf)   
            data = g.InitializeFields()
                    
            if i % size == rank:
                if pf["OutputDirectory"] != './':
                    try: os.mkdir("{0}".format(pf["OutputDirectory"]))
                    except OSError: pass        
                made = True
            else: 
                made = False
            
            # Wait here if parallelizing over grid
            if size > 1 and pf["ParallelizationMethod"] == 1: 
                MPI.COMM_WORLD.barrier()
    
        # Initialize integral tables
        if not pf['PhotonConserving']:
            iits = rtm.InitializeIntegralTables(pf, data)
            itabs = iits.TabulateRateIntegrals()        
            if pf["ExitAfterIntegralTabulation"]: 
                continue
        else:
            iits = itabs = None
                    
        # Initialize radiation and write data classes
        r = rtm.Radiate(pf, data, itabs, [iits.HIColumn, iits.HeIColumn, iits.HeIIColumn])
        w = rtm.WriteData(pf)
        
        # Compute initial timestep
        if IsRestart or pf["HIIRestrictedTimestep"] == 0: 
            dt = pf["CurrentTimestep"] * TimeUnits
        else:
            
            tau = 3 * [1.0]
            gamma = Beta = alpha = ncol = nion = 3 * [0]
            
            indices = None
            if pf['MultiSpecies'] > 0 and not pf['self.PhotonConserving']: 
                indices = self.coeff.Interpolate.GetIndices3D(ncol)
            
            if pf["PhotonConserving"]:
                Gamma = 0
                for i, E in enumerate(r.rs.DiscreteSpectrumSED):
                    Gamma += r.coeff.PhotoIonizationRate(E = E, Qdot = self.rs.Qdot[i], ncol = ncol,
                            n_HI = data['HIDensity'][-1], r = r.dx, dr = r.dx, dt = r.CellCrossingTime)
            else:                    
                Gamma = r.coeff.PhotoIonizationRate(species = 0, Lbol = r.rs.BolometricLuminosity(0), \
                    indices = indices, r = LengthUnits * StartRadius, 
                    nabs = [data['HIDensity'][-1], 0, 0], ncol = ncol)
                                                
            dt = r.control.ComputePhotonTimestep(tau, [Gamma, 0, 0], gamma, Beta, alpha, 
                [data['HIDensity'][-1], 0, 0], nion, ncol, 
                data['HIDensity'][-1] + data['HIIDensity'][-1], 
                data['HeIDensity'][-1] + data['HeIIDensity'][-1] + data['HeIIIDensity'][-1], 0)
                        
            # Play it safe if helium is involved
            if pf["MultiSpecies"]: 
                dt *= 0.1   
        
        # If (probalby for testing purposes) we have StopTime << 1, make sure dt <= StopTime        
        dt = min(dt, StopTime)
                    
        # Figure out data dump times, write out initial dataset (or not if this is a restart).
        ddt = np.arange(0, StopTime + dtDataDump, dtDataDump)
        t = pf["CurrentTime"] * TimeUnits
        h = dt
        wct = int(t / dtDataDump) + 1
        if not IsRestart: 
            if (pf["ParallelizationMethod"] == 1 and rank == 0) or \
               (pf["ParallelizationMethod"] == 2): 
                w.WriteAllData(data, 0, t, dt)
        
        # Wait for root processor to write out initial dataset if running in parallel            
        if size > 1 and pf["ParallelizationMethod"] == 1: 
            MPI.COMM_WORLD.barrier()
        
        # If we want to store timestep evolution, setup dump file
        if pf["OutputTimestep"]: 
            aw = 'w'
            if IsRestart: 
                aw = 'a'
            fdt = open('{0}/timestep_evolution.dat'.format(pf["OutputDirectory"]), aw)
            print >> fdt, '# t  dt'
            fdt.close()
        
        # Initialize load balance if running in parallel
        if size > 1: 
            lb = list(np.linspace(pf["GridDimensions"] / size, pf["GridDimensions"], size))
            lb.insert(0, 0)
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
            else: 
                write_now = False
    
            # Evolve photons
            data, h, newdt, lb = r.EvolvePhotons(data, t, dt, min(h, dt), lb)
            t += dt
            dt = newdt # dt for the next timestep
                                                                  
            # Write-out data                                        
            if write_now:
                wrote = False
                if (pf["ParallelizationMethod"] == 1 and rank == 0) or \
                   (pf["ParallelizationMethod"] == 2): 
                    w.WriteAllData(data, wct, tnow, dt)
                    wrote = True
                wct += 1
                
            if dt == 0: 
                raise ValueError('dt = 0.  Exiting.')  
               
            # Don't move on until root processor has written out data    
            if size > 1 and pf["ParallelizationMethod"] == 1: 
                MPI.COMM_WORLD.barrier()    
        
        if pf["OutputTimestep"]: 
            print >> fdt, ""
            fdt.close()
                    
        elapsed = time.time() - start  
        if i % size == rank:  
            print "Calculation {0} complete (output to {1}).  Elapsed time = {2} hours.".format(i + 1, pf["OutputDirectory"], round(elapsed / 3600., 2))
    
        f = open('{0}/RunComplete'.format(pf["OutputDirectory"]), 'w')
        print >> f, '# walltime (s)'
        print >> f, elapsed
        print >> f, ''
        f.close()
    
    print "Successful run. Exiting."

if __name__ == '__main__':
    
    # If restart or batch
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
    
    Shine(all_pfs, IsRestart = IsRestart)

