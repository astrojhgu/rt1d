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

FieldList = \
    ["Density", "Temperature", "HIDensity", "HIIDensity", "HeIDensity", \
     "HeIIDensity", "HeIIIDensity", "ElectronDensity"]    

m_e = 9.10938188*10**-28 		# Electron mass - [m_e] = g
m_p = 1.67262158*10**-24		# Proton mass - [m_p] = g
m_n = 1.67492729*10**-24        # Neutron mass - [m_n] = g

m_H = m_p + m_e
m_HeI = 2.0 * (m_p + m_n + m_e)
m_HeII = 2.0 * (m_p + m_n) + m_e                  

class InitializeGrid:
    def __init__(self, pf):
        self.Cosmology = Cosmology(pf)
        self.GridDimensions = pf["GridDimensions"]
        self.InitialRedshift = pf["InitialRedshift"]
        self.DensityProfile = pf["DensityProfile"]
        self.TemperatureProfile = pf["TemperatureProfile"]
        self.InitialTemperature = pf["InitialTemperature"]
        self.InitialHIIFraction = pf["InitialHIIFraction"]
        self.MultiSpecies = pf["MultiSpecies"]
        
        self.Y = 0.2477 * self.MultiSpecies
        self.DensityUnits = self.Cosmology.MeanBaryonDensity(self.InitialRedshift)
                        
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
                0: Uniform density given by cosmic mean at z = InitialRedshift.
                1: Density profile given by NFW model.  Requires r_s and c in this case too.
        """        
        
        if self.DensityProfile == 0: density = self.data * self.DensityUnits
        elif self.DensityProfile == 1: print 'NFW profile not yet implemented!'
            
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
        
        return (1.0 - self.Y) * (1.0 - self.InitialHIIFraction) * self.InitializeDensity() / m_H
    
    def InitializeHIIDensity(self):
        """
        Initialize ionized hydrogen density.
        """
        
        return (1.0 - self.Y) * self.InitialHIIFraction * self.InitializeDensity() / m_H
        
    def InitializeHeIDensity(self):
        """
        Initialize neutral helium density - initial ionized fraction of helium is assumed
        to be the same as that of hydrogen.
        """
        
        return self.Y * (1.0 - self.InitialHIIFraction) * self.InitializeDensity() / m_HeI
        
    def InitializeHeIIDensity(self):
        """
        Initialize ionized helium density - initial ionized fraction of helium is assumed
        to be the same as that of hydrogen.
        """
        
        return self.Y * self.InitialHIIFraction * self.InitializeDensity() / m_HeII
        
    def InitializeHeIIIDensity(self):
        """
        Initialize doubly ionized helium density - assumed to be zero.
        """
        
        return self.data * 0.0 
        
    def InitializeElectronDensity(self):
        """
        Initialize electron density - n_e = n_HII + n_HeII + 2n_HeIII (I'm pretty sure the equation in 
        Thomas and Zaroubi 2007 is wrong - they had n_e = n_HII + n_HeI + 2n_HeII).
        """
        
        return self.InitializeHIIDensity() + self.InitializeHeIIDensity() + 2.0 * self.InitializeHeIIIDensity()
        
        