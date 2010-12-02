"""
rtanl.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-12-01.

Description: General module for command line analysis of rt1d data.

Notes: Should be studying for comps...

How this works: from rtanl import * (put this in alias startup script as rta or something)
Then: rt = rtanl(parameter filename)

now we have an object containing all parameter info, use it to access datasets
     
In order for my shit to work, parameter sets must be labeled by the directory they live in.

Things we may want to do:
     -Once we have a simulation object, pull out data that has particular parameter combinations
     
Lingo: Master parameter file 'self.mpf': single dictionary containing a dictionary for each run that
was conducted in this set of runs.

    Dataset: all data dumps corresponding to a single parameter file
    Datadump: single data output, like 'dd0050.h5' for a single run.
     
want to be able to type ds.dd0001.T etc.     
     
"""

import os
import h5py
import numpy as np
import pylab as pl
from InitializeParameterSpace import *

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
        self.n_H = dd["HIDensity"].value + dd["HIIDensity"].value
        self.n_He = dd["HeIDensity"].value + dd["HeIIDensity"].value + dd["HeIIIDensity"].value
        
        self.n_e = dd["ElectronDensity"].value
        self.n_HI = dd["HIDensity"].value
        self.n_HII = dd["HIIDensity"].value
        self.n_HeI = dd["HeIDensity"].value
        self.n_HeII = dd["HeIIDensity"].value
        self.n_HeIII = dd["HeIIIDensity"].value
        
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
        """
        
        self.gd = gd        
                
        # Name of our parameter file
        self.pf = pf
                
        # Steal parameter space initialization from rt1d itself
        self.ips = InitializeParameterSpace(self.pf)
        
        # Create master parameter file (dict of all parameter files)
        self.mpf = self.ips.AllParameterSets()
             
    def LoadDataset(self, parvals = {}):
        """
        Return object containing access to all datadumps for this run.
        """
        
        # If we didn't specify parameters, it's probably because we're only interested 
        # in one dataset.  If we have access to more, let us know.
        if not parvals: 
            dsn = self.mpf.keys()[0]            # Just the name of the dataset - probably 'rt0000'
            pf = self.mpf[self.mpf.keys()[0]]   # Parameter file dictionary
           
        # Go find the dataset we were referring to.    
        else:
            
            # Count how many parameter files match 'parvals' - if > 1, degenerate
            degct = 0       
            
            for f in self.mpf.keys():
                
                # Number of parameters in 'parvals' that match this parameter file
                pct = 0     
                
                for par in self.mpf[f]:
                    if self.mpf[f][par] == parvals[par]: pct += 1
                    
                # If we found matches for each 'parval', we found our dataset    
                if pct == len(parvals): 
                    dsn = f
                    pf = self.mpf[ds]
                    
            if degct > 1: 
                print "There was more than one dataset matching the parameters you've supplied."
                print "We've loaded only the first - try again with a non-degenerate set of 'parvals'."
            
        # Now on to the hard part, load an object containing each data*dump* in this data*set*.
        
        # List all data*dumps* in this data*set*.
        alldds = os.listdir("{0}/{1}/".format(self.gd, dsn))
        
        ds = {}
        for ddf in alldds:
            f = h5py.File("{0}/{1}/{2}".format(self.gd, dsn, ddf))
            ID = ddf.partition('.')[0].strip('dd')
            if ID != '0000': ID = ID.lstrip('0')
            ds[int(ID)] = DataDump(f["Data"], f["ParameterFile"])
            f.close()
            
        return ds
            
            

           
                
                
                
        
        
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            