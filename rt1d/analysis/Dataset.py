"""
Dataset.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-12-01.

Description: General module for command line analysis of rt1d data.
     
"""

import os, re, h5py
import numpy as np
import pylab as pl
import rt1d.mods as rtm
from rt1d.analysis.DataDump import DataDump
        
class Dataset:
    def __init__(self, pf):
        """
        Initialize our analysis environment.  The variable 'dataset' can be either the master
        parameter file for a series of runs, or an individual parameter file for a single run.
        The parameter 'gd' is the global directory, which will default to the directory where
        we launched python from.
        
        Some jargon:
            'ds' = dataset, this is the dictionary we'll ultimately return
            'dsn' = dataset name, a string that is the name of the directory all datadumps live in
            'dd' = datadump, referring to the data in a specific time output in the entire dataset
            'ddf' = datadump file, just the filename of a particular dd, like 'dd0000.h5'
            
        Note: When supplying parameter file, if it is in the current directory, type './Filename' 
            so we get the path.
        
        """   
        
        # Global directory
        self.gd = os.getcwd()
                
        # Run directory
        self.rd = pf.rpartition('/')[0]        
                
        # Read in parameter file
        self.pf = rtm.ReadParameterFile(pf)
                        
        # Also need path to parameter file (not including the parameter file itself)
        self.od = self.pf["OutputDirectory"]
        
        self.data = self.load()
                                     
    def load(self):
        """
        Return object containing access to all datadumps for the run we've specified.
        """
        
        # List all data*dumps* in this data*set*.
        alldds = []
        for f in os.listdir("{0}/{1}".format(self.rd, self.od)):
            if not re.search('.h5', f): continue
            if not re.search('dd', f): continue # temporary hack
            alldds.append(f)
            
        ds = {}
        for ddf in alldds:
            f = h5py.File("{0}/{1}/{2}".format(self.rd, self.od, ddf))
            ID = ddf.partition('.')[0].strip('dd')
            ds[int(ID)] = DataDump(f["Data"], f["ParameterFile"])
            f.close()
            
        return ds

                        