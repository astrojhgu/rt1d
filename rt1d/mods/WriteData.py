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
from .Cosmology import Cosmology

try:
    import h5py
    h5 = True
except ImportError:
    print 'Module h5py not found. Will read/write to ASCII instead of HDF5.'
    h5 = False
    
try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    print "Module mpi4py not found.  No worries, we'll just run in serial."
    rank = 0
    size = 1

GlobalDir = os.getcwd()

class WriteData:
    def __init__(self, pf):
        self.pf = pf
        self.cosm = Cosmology(pf)
        self.OutputFormat = pf.OutputFormat and h5
        
    def WriteAllData(self, data, wct, t, dt):
        """
        Write all data to hdf5 file.
        """
        
        if self.OutputFormat == 0:
            self.WriteASCII(data, wct, t, dt)
        else:
            self.WriteHDF5(data, wct, t, dt)
            
        self.WriteParameterFile(wct, t, dt)    
        
    def WriteHDF5(self, data, wct, t, dt):
        """
        Write all data to hdf5 file.
        """
                                        
        f = h5py.File("{0}/{1}/{2}.h5".format(GlobalDir, self.pf.OutputDirectory.rstrip('/'), 
            self.GetDataDumpName(wct)), 'w') 

        pf_grp = f.create_group("ParameterFile")
        data_grp = f.create_group("Data")
        
        for par in self.pf.keys(): 
            if par == "CurrentTime": 
                pf_grp.create_dataset(par, data = t / self.pf.TimeUnits)
            elif par == "CurrentRedshift":
                pf_grp.create_dataset(par, data = self.cosm.TimeToRedshiftConverter(0, t, self.pf.InitialRedshift))
            elif par == "CurrentTimestep": 
                pf_grp.create_dataset(par, data = dt / self.pf.TimeUnits)
            else: 
                pf_grp.create_dataset(par, data = self.pf[par])
        
        for field in data: 
            if data[field].shape[0] > 0:
                data_grp.create_dataset(field, data = data[field])
        
        f.close()
        
        if rank == 0: 
            print "\nWrote {0}/{1}/{2}.h5\n".format(GlobalDir, self.pf.OutputDirectory.rstrip('/'), 
                self.GetDataDumpName(wct))

    def WriteASCII(self, data, wc, t, dt):
        """
        Write all data to ASCII file.
        """    
        
        raise ValueError('WriteASCII not yet implemented.')

    def GetDataDumpName(self, wct):
        """
        Return name of data dump to be written
        """
        
        return "{0}{1:04d}".format(self.pf.DataDumpName, wct)

    def WriteParameterFile(self, wct, t, dt):
        """
        Write out parameter file to ASCII format.
        """
                
        f = open("{0}/{1}/{2}".format(GlobalDir, self.pf.OutputDirectory, 
            self.GetDataDumpName(wct)), 'w')
        
        names = self.pf.keys()
        names.sort()
        
        print >> f, "{0} = {1}".format('ProblemType'.ljust(35, ' '), self.pf.ProblemType)
        
        for par in names:
            
            # ProblemType must be the first parameter
            if par == 'ProblemType': 
                continue
            
            if par == "CurrentTime": 
                val = t / self.pf.TimeUnits
            elif par == "CurrentRedshift":
                val = self.cosm.TimeToRedshiftConverter(0, t, self.pf.InitialRedshift)
            elif par == "CurrentTimestep": 
                val = dt / self.pf.TimeUnits
            else: 
                val = self.pf[par]
            
            print >> f, "{0} = {1}".format(par.ljust(35, ' '), val)
            
        f.close()    
        
        