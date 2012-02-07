"""
RT1D.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-10-14.

Description: Driver script for a 1D radiative transfer code.

"""

import sys, os, time, copy
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
    Initialize grid, radiation source, integral tables (maybe), and evolve
    radiation from CurrentTime to StopTime.
    """  
    
    # Convert parameter file to list of parameter files
    if type(pf) is not list:
        all_pfs = [pf]  
    else:
        all_pfs = pf    
        
    # Print some output to the screen        
    if rank == 0: 
        print "\nStarting rt1d...\n"
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
            return
              
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
            print "Box-crossing time = {0} (years), {1} (code)".format(LengthUnits / 29979245800.0 / \
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
        iits = rtm.InitializeIntegralTables(pf, data)
        if pf['TabulateIntegrals']:
            itabs = iits.TabulateRateIntegrals()        
            
            if pf['PhotonConserving'] and pf['AutoFallback']:
                itabs = [itabs]
                tmp_pf = copy.deepcopy(pf)
                tmp_pf['PhotonConserving'] = 0
                tmp_iits = rtm.InitializeIntegralTables(tmp_pf, data)
                itabs.append(tmp_iits.TabulateRateIntegrals())
            
            if pf["ExitAfterIntegralTabulation"]: 
                continue
        else:
            print '\nNo integral tabulation necessary!'
            itabs = None
            if IsRestart:
                print '\n'
                    
        # Initialize radiation and write data classes
        r = rtm.Radiate(pf, data, itabs, [iits.HIColumn, iits.HeIColumn, iits.HeIIColumn])
        w = rtm.WriteData(pf)
        
        # Compute initial timestep
        if IsRestart or pf["HIRestrictedTimestep"] == 0: 
            dt = pf["CurrentTimestep"] * TimeUnits
        elif pf["InitialTimestep"] > 0:
            dt = pf["InitialTimestep"] * TimeUnits
        else:
            # CONDENSE THIS
            tau1 = ncol = np.array(3 * [1.0])
            tau0 = gamma = Beta = alpha = nion = xi = np.array(3 * [0])
            nabs = np.array([data['HIDensity'][0], data['HeIDensity'][0], data['HeIIDensity'][0]])
            n_H = data['HIDensity'][0] + data['HIIDensity'][0]
            n_He = data['HeIDensity'][0] + data['HeIIDensity'][0] + data['HeIIIDensity'][0]
            n_e = data['HIIDensity'][0] + data['HeIIDensity'][0] + 2. * data['HeIIIDensity'][0]            
                        
            indices = None
            if pf['MultiSpecies'] > 0 and pf['TabulateIntegrals']: 
                indices = r.coeff.Interpolate.GetIndices3D(ncol)            
                
            Gamma = np.zeros(3)                                    
            if itabs is None:
                for i, E in enumerate(r.rs.E):
                    Gamma[0] += r.coeff.PhotoIonizationRate(E = E, Qdot = r.rs.IonizingPhotonLuminosity(i = i), \
                        ncol = ncol, nabs = nabs, r = LengthUnits * StartRadius, \
                        dr = r.dx[0], species = 0, tau = tau0, Lbol = r.rs.BolometricLuminosity(0))
                    
                    if pf['MultiSpecies']:
                        Gamma[1] += r.coeff.PhotoIonizationRate(E = E, Qdot = r.rs.IonizingPhotonLuminosity(i = i), \
                            ncol = ncol, nabs = nabs, r = LengthUnits * StartRadius, \
                            dr = r.dx[0], species = 1, tau = tau0, Lbol = r.rs.BolometricLuminosity(0))
                        Gamma[2] += r.coeff.PhotoIonizationRate(E = E, Qdot = r.rs.IonizingPhotonLuminosity(i = i), \
                            ncol = ncol, nabs = nabs, r = LengthUnits * StartRadius, \
                            dr = r.dx[0], species = 2, tau = tau0, Lbol = r.rs.BolometricLuminosity(0))    
                        
            else:                    
                Gamma[0] = r.coeff.PhotoIonizationRate(species = 0, Lbol = r.rs.BolometricLuminosity(0), \
                    indices = indices, r = LengthUnits * StartRadius, dr = r.dx[0],
                    nabs = nabs, ncol = ncol, tau = tau0)
                    
                if pf['MultiSpecies']:
                    Gamma[1] = r.coeff.PhotoIonizationRate(species = 1, Lbol = r.rs.BolometricLuminosity(0), \
                        indices = indices, r = LengthUnits * StartRadius, dr = r.dx[0],
                        nabs = nabs, ncol = ncol, tau = tau0)
                    Gamma[2] = r.coeff.PhotoIonizationRate(species = 2, Lbol = r.rs.BolometricLuminosity(0), \
                        indices = indices, r = LengthUnits * StartRadius, dr = r.dx[0],
                        nabs = nabs, ncol = ncol, tau = tau0)                            
                        
            dt = r.control.ComputeInitialPhotonTimestep(tau1, nabs, nion, ncol, 
                n_H, n_He, n_e, n_H + n_He + n_e, Gamma, gamma, Beta, alpha, xi)
                                                                                                                              
        # If (probalby for testing purposes) we have StopTime << 1, make sure dt <= StopTime        
        dt = min(dt, StopTime)
                    
        # Figure out data dump times, write out initial dataset (or not, if this is a restart).
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
        
        # SOLVE RADIATIVE TRANSFER              
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
                
            # Raise error if funny stuff happening    
            if dt < 0: 
                raise ValueError('ERROR: dt < 0.  Exiting.') 
            elif dt == 0:
                raise ValueError('ERROR: dt = 0.  Exiting.')  
            elif np.isnan(dt):  
                raise ValueError('ERROR: dt -> inf.  Exiting.')    
               
            # Don't move on until root processor has written out data    
            if size > 1 and pf["ParallelizationMethod"] == 1: 
                MPI.COMM_WORLD.barrier()    
        
        # Close timestep file
        if pf["OutputTimestep"]: 
            print >> fdt, ""
            fdt.close()
                    
        elapsed = time.time() - start  
        if i % size == rank:  
            print "Calculation {0} complete (output to {1}).  Elapsed time = {2} hours.".format(i + 1, pf["OutputDirectory"], round(elapsed / 3600., 4))
    
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

