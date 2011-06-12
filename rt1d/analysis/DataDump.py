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
        
        self.GridDimensions = pf["GridDimensions"].value
        self.grid = np.arange(self.GridDimensions)[1:]
        
        self.r = np.arange(self.GridDimensions - 1) * pf["LengthUnits"].value / self.GridDimensions
        self.t = pf["CurrentTime"].value * pf["TimeUnits"].value
        self.dx = pf["LengthUnits"].value / self.GridDimensions
        
        # Fields
        self.T = dd["Temperature"].value[1:]
        self.n_e = dd["ElectronDensity"].value[1:]
        self.n_HI = dd["HIDensity"].value[1:]
        self.n_HII = dd["HIIDensity"].value[1:]
        self.n_HeI = dd["HeIDensity"].value[1:]
        self.n_HeII = dd["HeIIDensity"].value[1:]
        self.n_HeIII = dd["HeIIIDensity"].value[1:]
        
        self.n_H = self.n_HI + self.n_HII
        self.n_He = self.n_HeI + self.n_HeII + self.n_HeIII
        
        self.x_HI = self.n_HI / self.n_H
        self.x_HII = self.n_HII / self.n_H
        self.x_HeI = self.n_HeI / self.n_He
        self.x_HeII = self.n_HeII / self.n_He
        self.x_HeIII = self.n_HeIII / self.n_He
                
        # Column densities
        self.ncol_HI = np.cumsum(self.n_HI) * self.dx 
        self.ncol_HeI = np.cumsum(self.n_HeI) * self.dx
        self.ncol_HeII = np.cumsum(self.n_HeII) * self.dx