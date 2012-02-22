"""
DataDump.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-12-01.

Description: Data dump object.

Notes: 
     
"""

import numpy as np

neglible_column = 1.

class DataDump:
    def __init__(self, dd, pf):
        """
        Turns an hdf5 file object into attributes of the DataDump object!
        
        Note: pf is an hdf5 group object here.
        
        """        
        
        self.LengthUnits = pf["LengthUnits"].value
        self.StartRadius = pf["StartRadius"].value
        self.GridDimensions = pf["GridDimensions"].value
        self.grid = np.arange(self.GridDimensions)
        
        # Deal with log-grid
        if pf["LogarithmicGrid"].value:
            self.r = np.logspace(np.log10(self.StartRadius * self.LengthUnits), \
                np.log10(self.LengthUnits), self.GridDimensions + 1)
        else:
            rmin = self.StartRadius * self.LengthUnits
            self.r = np.linspace(rmin, self.LengthUnits, self.GridDimensions + 1)
        
        self.dx = np.diff(self.r)   
        self.r = self.r[0:-1] 
                
        # Shift radii to cell-centered values
        self.r += self.dx / 2.   
                            
        self.t = pf["CurrentTime"].value * pf["TimeUnits"].value
        
        # Fields
        self.T = dd["Temperature"].value
        self.n_e = dd["ElectronDensity"].value
        self.n_HI = dd["HIDensity"].value
        self.n_HII = dd["HIIDensity"].value
        self.n_H = self.n_HI + self.n_HII
        self.x_HI = self.n_HI / self.n_H
        self.x_HII = self.n_HII / self.n_H
        self.ncol_HI = np.roll(np.cumsum(self.n_HI * self.dx), 1)
        self.ncol_HI[0] = neglible_column
        self.dtPhoton = dd["dtPhoton"].value / pf["TimeUnits"].value
        self.n_B = self.n_H + self.n_e
        
        self.n_HeI = dd["HeIDensity"].value
        self.n_HeII = dd["HeIIDensity"].value
        self.n_HeIII = dd["HeIIIDensity"].value
        self.n_He = self.n_HeI + self.n_HeII + self.n_HeIII
        self.x_HeI = self.n_HeI / self.n_He
        self.x_HeII = self.n_HeII / self.n_He
        self.x_HeIII = self.n_HeIII / self.n_He
        self.ncol_HeI = np.roll(np.cumsum(self.n_HeI * self.dx), 1)
        self.ncol_HeII = np.roll(np.cumsum(self.n_HeII * self.dx), 1)
        self.ncol_e = np.roll(np.cumsum(self.n_e * self.dx), 1)
        self.ncol_HeI[0] = self.ncol_HeII[0] = neglible_column
        self.n_B += self.n_He
        self.f_e = self.n_e / self.n_B    
        
        self.Gamma = np.zeros([3, self.GridDimensions])
        self.k_H = np.zeros([3, self.GridDimensions])
        self.gamma = np.zeros([3, self.GridDimensions])
        
        if pf["OutputRates"].value:
            for i in xrange(3):
                self.Gamma[i] = dd['PhotoIonizationRate%i' % i].value
                self.k_H[i] = dd['PhotoHeatingRate%i' % i].value
                self.gamma[i] = dd['SecondaryIonizationRate%i' % i].value
                
        
    def __getitem__(self, name):
        """
        Get data by name we use in hdf5 storage.
        """    
        
        if name == 'HIDensity':
            return self.n_HI
        elif name == 'HIIDensity':
            return self.n_HII
        elif name == 'HIIDensity':
            return self.n_HII
        elif name == 'HeIDensity':
            return self.n_HeI
        elif name == 'HeIIDensity':
            return self.n_HeII
        elif name == 'HeIIIDensity':
            return self.n_HeIII
                    
            
                    