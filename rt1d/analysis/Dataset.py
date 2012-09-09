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
from .DataDump import DataDump
from ..mods.ReadParameterFile import ReadParameterFile
        
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
        
        """   
        
        # Global directory
        self.gd = os.getcwd()
                
        # Run directory
        self.rd = pf.rpartition('/')[0]
        
        if not self.rd.strip():
            self.rd = '.'        
                
        # Read in parameter file
        self.pf = ReadParameterFile(pf)
                        
        # Also need path to parameter file (not including the parameter file itself)
        # NOTE: THIS IS RELATIVE TO RUN DIRECTORY
        self.od = self.pf['OutputDirectory']
        
        # Path to all output - ABSOLUTE PATH
        self.path_to_output = '%s/%s/%s' % (self.gd, self.rd, self.od)
        self.path_to_output = self.path_to_output.strip().rstrip('.')
        
        self.data = self.load()
        self.t, self.dt = self.read_timestep_evolution()
                                     
    def load(self):
        """
        Return object containing access to all datadumps for the run we've specified.
        """
        
        # List all data*dumps* in this data*set*.
        alldds = []
        for f in os.listdir(self.path_to_output):
            if not re.search('.h5', f): 
                continue
            if not re.search('dd', f): 
                continue # temporary hack
            alldds.append(f)
            
        ds = {}
        for ddf in alldds:
            f = h5py.File("%s/%s" % (self.path_to_output, ddf))
            ID = ddf.partition('.')[0].strip('dd')
            try:
                ds[int(ID)] = DataDump(f["data"], f["parameters"])
            except KeyError: # backwards compatibility
                try:
                    ds[int(ID)] = DataDump(f["Data"], f["ParameterFile"])
                except KeyError:
                    print 'Data in %s likely corrupted.' % ddf
            f.close()
            
        return ds

    def read_timestep_evolution(self):
        """
        Read in t vs. dt. 
        """
        
        if not self.pf['OutputTimestep']:
            return None, None
        
        if not os.path.exists('%s/timestep_evolution.dat' % self.path_to_output):
            return None, None
            
        f = open('%s/timestep_evolution.dat' % self.path_to_output, 'r')
        t = []
        dt = []
        for line in f:
            if not line.strip():
                continue
            if line[0] == '#':
                continue
            
            newline = line.split()
            t.append(float(newline[0]))
            dt.append(float(newline[1]))
            
        f.close()
        
        return np.array(t), np.array(dt)      
        
            
                        