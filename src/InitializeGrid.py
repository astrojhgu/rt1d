"""
InitializeGrid.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-10-18.

Description: Construct 1D arrays for our data in code units.  Return as dictionary.

Notes: Do I want to keep track of CellWidth as well?
     
"""

import numpy as np
from Cosmology import *
from FieldList import *

Y = 0.2477      # Primordial helium abundance by mass                      

class InitializeGrid:
    def __init__(self, pf):
        self.Cosmology = Cosmology(pf)
        self.GridDimensions = pf["GridDimensions"]
        self.InitialRedshift = pf["InitialRedshift"]
        self.DensityProfile = pf["DensityProfile"]
        self.TemperatureProfile = pf["TemperatureProfile"]
        self.InitialTemperature = pf["InitialTemperature"]
        self.InitialHIIFraction = pf["InitialHIIFraction"]
        
        if self.DensityProfile == 1: self.DensityUnits = self.Cosmology.MeanDensity(self.InitialRedshift)
        else: self.DensityUnits = pf["DensityUnits"]                
                        
        # Generic data array                
        self.data = np.ones(self.GridDimensions)
                    
    def InitializeFields(self):
        """
        Return dictionary of all fields.
        """
        
        fields = {}
        for field in FieldList:
            fields[field] = eval("self.Initialize{0}()".format(field))
            
        return fields
            
    def InitializeDensity(self):
        """
        Initialize the gas density - depends on parameter DensityProfile as follows:
        
            DensityProfile:
                0: Uniform density given by DensityUnits.
                1: Uniform density given by cosmic mean at z = InitialRedshift.
                2: Density profile given by NFW model.  Requires r_s and c in this case too.
        """        
        
        if self.DensityProfile < 2: density = self.data * self.DensityUnits
        elif self.DensityProfile == 2: print 'NFW profile not yet implemented!'
            
        return density
        
    def InitializeTemperature(self):
        """
        Initialize temperature - depends on parameter TemperatureProfile as follows:
        
            TemperatureProfile:
                0: Uniform temperature given by InitialTemperature
        """
        
        if self.TemperatureProfile == 0: temperature = self.data * self.InitialTemperature 
        
        return temperature
        
    def InitializeHIDensity(self):
        """
        Initialize neutral hydrogen density.
        """
        
        return (1.0 - Y) * (1.0 - self.InitialHIIFraction) * self.InitializeDensity()
    
    def InitializeHIIDensity(self):
        """
        Initialize ionized hydrogen density.
        """
        
        return (1.0 - Y) * self.InitialHIIFraction * self.InitializeDensity()   
        
    def InitializeHeIDensity(self):
        """
        Initialize neutral helium density - initial ionized fraction of helium is assumed
        to be the same as that of hydrogen.
        """
        
        return Y * (1.0 - self.InitialHIIFraction) * self.InitializeDensity() 
        
    def InitializeHeIIDensity(self):
        """
        Initialize ionized helium density - initial ionized fraction of helium is assumed
        to be the same as that of hydrogen.
        """
        
        return Y * self.InitialHIIFraction * self.InitializeDensity() 
        
    def InitializeHeIIIDensity(self):
        """
        Initialize doubly ionized helium density - assumed to be zero.
        """
        
        return self.data * 0.0 
        
        
        