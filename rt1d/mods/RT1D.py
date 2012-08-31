"""
RT1D.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-10-14.

Description: Driver script for a 1D radiative transfer code.

"""

import sys, os, time, copy, shutil
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
    
def Shine(pf, r = None, data = None, IsRestart = False):
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
        if pf['ParallelizationMethod'] == 2:
            if i % size != rank: continue
            
        # Check for parameter conflicts    
        conflicts, errmsg = rtm.CheckForParameterConflicts(pf)
        if conflicts:
            print 'ERROR -- PARAMETER VALUES IN CONFLICT:'
            for msg in errmsg:
                print msg
            print '\n'    
            return
            
        # Initialize grid and file system
        print 'Initializing grid...'
        if data is not None:
            data = data
            g = rtm.InitializeGrid(pf)
        if IsRestart: 
            data = ICs
            g = rtm.InitializeGrid(pf)
        else:
            g = rtm.InitializeGrid(pf)   
            data = g.InitializeFields()
                    
            if i % size == rank:
                if pf['OutputDirectory'] != '.':
                    try: 
                        os.mkdir(pf['OutputDirectory'])
                    except OSError: 
                        pass
            
        # Wait here if parallelizing over grid
        if size > 1 and pf['ParallelizationMethod'] == 1: 
            MPI.COMM_WORLD.barrier()
          
        # Initialize radiation source class(es)    
        print 'Initializing radiation source...'
        rs = rtm.RadiationSources(pf)
        
        # Copy a few things to OutputDirectory    
        if pf['OutputDirectory'] != '.':
        
            # Copy source files to OutputDirectory
            if pf['SourceFiles'] != 'None':
                for fn in pf['SourceFiles']:
                    shutil.copy(fn, pf['OutputDirectory'])
            
            # Copy spectrum files to OutputDirectory
            for src in rs.all_sources:
                if src.pf['SpectrumFile'] == 'None':
                    continue
                shutil.copy(src.pf['SpectrumFile'], pf['OutputDirectory'])
            
        # Initialize radiation and write data classes
        r = rtm.Radiate(pf, rs, g)
        w = rtm.WriteData(pf)
        
        # Compute initial timestep
        if IsRestart or pf['HIRestrictedTimestep'] == 0: 
            dt = pf['CurrentTimestep'] * pf['TimeUnits']
        elif pf['InitialTimestep'] > 0:
            dt = pf['InitialTimestep'] * pf['TimeUnits']
        else:
            dt = r.control.ComputeInitialPhotonTimestep(data, r)
                                                                                                                
        # If (probably for testing purposes) we have StopTime << 1, make sure dt <= StopTime1
        dt = min(dt, pf['StopTime'] * pf['TimeUnits'])
        
        # Figure out data dump times, write out initial dataset (or not, if this is a restart).
        ddt = np.linspace(0, pf['StopTime'] * pf['TimeUnits'], 1. + pf['StopTime'] / pf['dtDataDump'])    

        if pf['LogarithmicDataDump']:
            dlogdt = np.logspace(np.log10(pf['InitialLogDataDump'] * pf['TimeUnits']), 
                np.log10(pf['StopTime'] * pf['TimeUnits']), pf['NlogDataDumps'])
            ddt = np.concatenate((ddt, dlogdt))
            ddt.sort()    
        
        h = dt                    
        t = pf['CurrentTime'] * pf['TimeUnits']
        wct = np.argmin(np.abs(t - ddt)) + 1
        if not IsRestart: 
            if  pf['ParallelizationMethod'] == 0 or \
               (pf['ParallelizationMethod'] == 1 and rank == 0) or \
               (pf['ParallelizationMethod'] == 2): 
                w.WriteAllData(data, 0, t, dt)
        
        # Wait for root processor to write out initial dataset if running in parallel            
        if size > 1 and pf['ParallelizationMethod'] == 1: 
            MPI.COMM_WORLD.barrier()
        
        # If we want to store timestep evolution, setup dump file
        if pf['OutputTimestep']: 
            aw = 'w'
            if IsRestart: 
                aw = 'a'
            fdt = open('%s/timestep_evolution.dat' % pf['OutputDirectory'], aw)
            print >> fdt, '# t  dt'
            fdt.close()
        
        # Initialize load balance if running in parallel
        if size > 1: 
            lb = list(np.linspace(pf['GridDimensions'] / size, pf['GridDimensions'], size))
            lb.insert(0, 0)
        else: 
            lb = None                
                    
        # SOLVE RADIATIVE TRANSFER              
        while t < (pf['StopTime'] * pf['TimeUnits']):
                                    
            if pf['OutputTimestep']: 
                fdt = open('%s/timestep_evolution.dat' % pf['OutputDirectory'], 'a')
                print >> fdt, t / pf['TimeUnits'], dt / pf['TimeUnits']
                                                                        
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
            
            # dt for the next timestep
            dt = min(newdt, pf['MaximumGlobalTimestep'] * pf['TimeUnits']) 
                                                                              
            # Write-out data                                        
            if write_now:
                wrote = False
                if  pf['ParallelizationMethod'] == 0 or \
                   (pf['ParallelizationMethod'] == 1 and rank == 0) or \
                   (pf['ParallelizationMethod'] == 2): 
                    w.WriteAllData(data, wct, tnow, dt)
                    wrote = True
                wct += 1
                
            # Raise error if any funny stuff happens    
            if dt < 0: 
                raise ValueError('ERROR: dt < 0.  Exiting.') 
            elif dt == 0:
                raise ValueError('ERROR: dt = 0.  Exiting.')  
            elif np.isnan(dt):  
                raise ValueError('ERROR: dt -> inf.  Exiting.')    
               
            # Don't move on until root processor has written out data    
            if size > 1 and pf['ParallelizationMethod'] == 1: 
                MPI.COMM_WORLD.barrier()    
        
        # Close timestep file
        if pf['OutputTimestep']: 
            print >> fdt, ""
            fdt.close()
                    
        elapsed = time.time() - start  
        if i % size == rank:  
            print "Calculation %i complete (output to %s).  Elapsed time = %g minutes." % (i + 1, 
                pf['OutputDirectory'], round(elapsed / 60., 3))
    
        f = open('%s/RunComplete' % pf['OutputDirectory'], 'w')
        print >> f, '# walltime (s) date'
        print >> f, elapsed, time.ctime()
        print >> f, ''
        f.close()
            
    if __name__ != '__main__':
        return data, dt
    else:
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

