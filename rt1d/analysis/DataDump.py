"""
DataDump.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-12-01.

Description: Data dump object.

Notes: 
     
"""

import numpy as np

class DataDump:
    def __init__(self, dd, pf):
        """
        Turns an hdf5 file object into attributes of the DataDump object!
        
        Note: I'm not including the first cells because values will be zero and mess
        up log plots!
        
        """        
        
        self.LengthUnits = pf["LengthUnits"].value
        self.StartRadius = pf["StartRadius"].value
        self.GridDimensions = pf["GridDimensions"].value
        
        # Deal with log-grid
        if pf['LogarithmicGrid']:
            self.r = np.logspace(np.log10(self.StartRadius * self.LengthUnits), \
                np.log10(self.LengthUnits), self.GridDimensions)
            r_tmp = np.concatenate([[0], self.r])
            self.dx = np.diff(r_tmp)    
            self.grid = np.arange(len(self.r))            
        else:
            self.dx = self.LengthUnits / self.GridDimensions
            rmin = max(self.dx, self.StartRadius * self.LengthUnits)
            self.r = np.linspace(rmin, self.LengthUnits, self.GridDimensions)
            self.grid = np.arange(len(self.r))    
                    
        self.t = pf["CurrentTime"].value * pf["TimeUnits"].value
        
        # Fields
        self.T = dd["Temperature"].value
        self.n_e = dd["ElectronDensity"].value
        self.n_HI = dd["HIDensity"].value
        self.n_HII = dd["HIIDensity"].value
        self.n_H = self.n_HI + self.n_HII
        self.x_HI = self.n_HI / self.n_H
        self.x_HII = self.n_HII / self.n_H
        self.ncol_HI = np.cumsum(self.n_HI * self.dx)
        self.dtPhoton = dd["dtPhoton"].value / pf["TimeUnits"].value
        
        #try:
        #    if not pf["InfiniteSpeedOfLight"]: self.PhotonPackages = dd["PhotonPackages"].value
        #except KeyError: pass
        
        if pf["MultiSpecies"].value > 0:
            self.n_HeI = dd["HeIDensity"].value
            self.n_HeII = dd["HeIIDensity"].value
            self.n_HeIII = dd["HeIIIDensity"].value
            self.n_He = self.n_HeI + self.n_HeII + self.n_HeIII
            self.x_HeI = self.n_HeI / self.n_He
            self.x_HeII = self.n_HeII / self.n_He
            self.x_HeIII = self.n_HeIII / self.n_He
            self.ncol_HeI = np.cumsum(self.n_HeI * self.dx)
            self.ncol_HeII = np.cumsum(self.n_HeII * self.dx)
            self.ncol_e = np.cumsum(self.n_e * self.dx)
            
            