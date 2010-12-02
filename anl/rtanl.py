"""
rtanl.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-12-01.

Description: General module for command line analysis of rt1d data.

Notes: Should be studying for comps...

Brief Tutorial:
    from rtanl import *
    rt = rtanl('ParameterFile.dat')
    sim = rt.LoadDataset()
    
    T = sim[0].T    # Pulls out temperature field from dd0001!
     
"""

import os, re, h5py
import numpy as np
import pylab as pl
from InitializeParameterSpace import *
from SetDefaultParameterValues import *

class DataDump:
    def __init__(self, dd, pf):
        """
        Turns an hdf5 file object into attributes of the DataDump object!
        """
        
        self.GridDimensions = pf["GridDimensions"].value
        self.r = np.arange(self.GridDimensions) * pf["LengthUnits"].value / self.GridDimensions
        self.t = pf["CurrentTime"].value * pf["TimeUnits"].value
        
        # Fields
        self.T = dd["Temperature"].value
        self.n_e = dd["ElectronDensity"].value
        self.n_HI = dd["HIDensity"].value
        self.n_HII = dd["HIIDensity"].value
        self.n_HeI = dd["HeIDensity"].value
        self.n_HeII = dd["HeIIDensity"].value
        self.n_HeIII = dd["HeIIIDensity"].value
        
        self.n_H = n_HI + n_HII
        self.n_He = n_HeI + n_HeII + n_HeIII
        
        self.x_HI = self.n_HI / self.n_H
        self.x_HII = self.n_HII / self.n_H
        self.x_HeI = self.n_HeI / self.n_He
        self.x_HeII = self.n_HeII / self.n_He
        self.x_HeIII = self.n_HeIII / self.n_He
        
class rtanl:
    def __init__(self, pf, gd = os.getcwd()):
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
        self.gd = gd       
                
        # Name of our parameter file (including path to it)
        self.pf = pf        
                
        # Also need path to parameter file (not including the parameter file itself)
        self.sd = self.pf.rsplit('/')[0]       
                
        # Steal parameter space initialization from rt1d itself
        self.ips = InitializeParameterSpace(self.pf)
        
        # Create master parameter file (dict of all parameter files)
        self.mpf = self.ips.AllParameterSets()
             
    def load(self, parvals = {}, filename = None):
        """
        Return object containing access to all datadumps for the run we've specified.  Can find run
        by particular set of parameters or by its name if we happen to know it.
        """
        
        # If we didn't specify parameters, it's probably because we're only interested 
        # in one dataset.  If we have access to more, let us know.
        if not parvals: 
            if filename is not None: 
                dsn = filename
                pf = self.mpf[dsn]
            else:
                dsn = self.mpf.keys()[0]            # Just the name of the dataset - probably 'rt0000'
                pf = self.mpf[self.mpf.keys()[0]]   # Parameter file dictionary
           
        # Go find the dataset we were referring to.    
        else:
            
            # Count how many parameter files match 'parvals' - if > 1, degenerate
            degct = 0
            
            for f in self.mpf.keys():
                
                # Number of parameters in 'parvals' that match this parameter file
                pct = 0     
                                
                for par in parvals:
                    if self.mpf[f][par] == parvals[par]: pct += 1
                                    
                # If we found matches for each 'parval', we found our dataset    
                if pct == len(parvals): 
                    dsn = f
                    pf = self.mpf[dsn]
                    
            if degct > 1: 
                print "There was more than one dataset matching the parameters you've supplied."
                print "We've loaded only the first - try again with a non-degenerate set of 'parvals'."
            
        # Now on to the hard part, load an object containing each data*dump* in this data*set*.
                
        # List all data*dumps* in this data*set*.
        alldds = []
        for f in os.listdir("{0}/{1}/{2}".format(self.gd, self.sd, dsn)):
            if not re.search('.h5', f): continue
            alldds.append(f)
            
        
        
        ds = {}
        for ddf in alldds:
            f = h5py.File("{0}/{1}/{2}/{3}".format(self.gd, self.sd, dsn, ddf))
            ID = ddf.partition('.')[0].strip('dd')
            ds[int(ID)] = DataDump(f["Data"], f["ParameterFile"])
            f.close()
            
        return ds
            
            

           
                
                
                
        
        
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            