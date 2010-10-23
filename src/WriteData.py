"""
WriteData.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-08-18.

Description: Write out data to text or hdf5 files.  We will decide based on the value
of the parameter 'WriteFormat' in the parameter file.
     
"""

import h5py, os
import numpy as np

GlobalDir = os.getcwd()

class WriteData:
    def __init__(self, pf):
        self.pf = pf
        
    def WriteAllData(self, data, prefix, wct, t):
        DataDumpName = "{0}{1:04d}".format(self.pf["DataDumpName"], wct)
                
        f = h5py.File("{0}/{1}/{2}.h5".format(GlobalDir, prefix, DataDumpName), 'w') 

        pf_grp = f.create_group("ParameterFile")
        data_grp = f.create_group("Data")
        
        pf_grp.create_dataset("CurrentTime", data = t)
        for par in self.pf: pf_grp.create_dataset(par, data = self.pf[par])
        for field in data: data_grp.create_dataset(field, data = data[field])
        
        f.close()
        
        
        
