"""
WriteData.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-10-18.

Description: Write out data to text or hdf5 files.  We will decide based on the value
of the parameter 'WriteFormat' in the parameter file.
     
"""

import h5py, os
import numpy as np

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except:
    ImportError("Module mpi4py not found.  No worries, we'll just run in serial.")
    rank = 0
    size = 1

GlobalDir = os.getcwd()

class WriteData:
    def __init__(self, pf):
        self.pf = pf
        self.TimeUnits = pf["TimeUnits"]
        self.OutputDirectory = pf["OutputDirectory"]

    def WriteAllData(self, data, wct, t, dt):
        DataDumpName = "{0}{1:04d}".format(self.pf["DataDumpName"], wct)
                                
        f = h5py.File("{0}/{1}/{2}.h5".format(GlobalDir, self.OutputDirectory, DataDumpName), 'w') 

        pf_grp = f.create_group("ParameterFile")
        data_grp = f.create_group("Data")
                        
        for par in self.pf: 
            if par == "CurrentTime": pf_grp.create_dataset(par, data = t / self.TimeUnits)
            elif par == "CurrentTimestep": pf_grp.create_dataset(par, data = dt / self.TimeUnits)
            else: pf_grp.create_dataset(par, data = self.pf[par])
        for field in data: 
            if len(data[field]) == 0: continue
            data_grp.create_dataset(field, data = data[field])
        
        f.close()
        
        if rank == 0: print "Wrote {0}/{1}/{2}.h5\n".format(GlobalDir, self.OutputDirectory, DataDumpName)
        
        self.WriteParameterFile(wct, t, dt)

    def WriteParameterFile(self, wct, t, dt):
        """
        Write out parameter file to ASCII format.
        """
        
        DataDumpName = "{0}{1:04d}".format(self.pf["DataDumpName"], wct)
        
        f = open("{0}/{1}/{2}".format(GlobalDir, self.OutputDirectory, DataDumpName), 'w')
        
        names = self.pf.keys()
        names.sort()
        
        print >> f, "{0} = {1}".format('ProblemType'.ljust(35, ' '), self.pf['ProblemType'])
        
        for par in names:
            
            # ProblemType must be the first parameter
            if par == 'ProblemType': continue
            
            if par == "CurrentTime": val = t / self.TimeUnits
            elif par == "CurrentTimestep": val = dt / self.TimeUnits
            else: val = self.pf[par]
            
            print >> f, "{0} = {1}".format(par.ljust(35, ' '), val)
            
        f.close()    
        
        