"""
InitializeGrid.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-10-18.

Description: Construct 1D arrays for our data in code units.  Return as dictionary.

Notes: Non-constant profiles need fixing.
     
"""

import numpy as np
from Cosmology import *

FieldList = \
    ["Temperature", "HIDensity", "HIIDensity", "HeIDensity", \
     "HeIIDensity", "HeIIIDensity", "ElectronDensity"]    

m_e = 9.10938188*10**-28 		# Electron mass - [m_e] = g
m_p = 1.67262158*10**-24		# Proton mass - [m_p] = g
m_n = 1.67492729*10**-24        # Neutron mass - [m_n] = g
m_H = m_p + m_e        

tiny_number = 1e-12          

class InitializeGrid:
    def __init__(self, pf):
        self.pf = pf
        self.Cosmology = Cosmology(pf)
        self.GridDimensions = int(pf["GridDimensions"])
        self.LogGrid = pf["LogarithmicGrid"]
        self.StartRadius = pf["StartRadius"]
        self.InitialRedshift = pf["InitialRedshift"]
        self.DensityProfile = pf["DensityProfile"]
        self.InitialDensity = pf["InitialDensity"]
        self.TemperatureProfile = pf["TemperatureProfile"]
        self.InitialTemperature = pf["InitialTemperature"]
        self.IonizationProfile = pf["IonizationProfile"]
        self.InitialHIIFraction = pf["InitialHIIFraction"]
        self.MultiSpecies = pf["MultiSpecies"]
        self.MinimumSpeciesFraction = pf["MinimumSpeciesFraction"]
        self.LengthUnits = pf["LengthUnits"]
        self.OutputRates = pf["OutputRates"]
        
        self.Clump = pf["Clump"]
        if self.Clump:
            self.ClumpPosition = pf["ClumpPosition"] * self.GridDimensions
            self.ClumpOverdensity = pf["ClumpOverdensity"]
            self.ClumpRadius = pf["ClumpRadius"] * self.GridDimensions
            self.ClumpTemperature = pf["ClumpTemperature"]
            self.ClumpDensityProfile = pf["ClumpDensityProfile"]
        
        self.Y = 0.2477 * self.MultiSpecies
                
        # Deal with log-grid
        if self.LogGrid:
            self.r = np.logspace(np.log10(self.StartRadius * self.LengthUnits), \
                np.log10(self.LengthUnits), self.GridDimensions)
            r_tmp = np.concatenate([[0], self.r])
            self.dx = np.diff(r_tmp)    
        else:
            self.dx = self.LengthUnits / self.GridDimensions
            rmin = max(self.dx, self.StartRadius * self.LengthUnits)
            self.r = np.linspace(rmin, self.LengthUnits, self.GridDimensions)
        
        self.grid = np.arange(len(self.r))
                            
        # Generic data array                
        self.density = map(self.InitializeDensity, self.grid)
        self.ionization = map(self.InitializeIonization, self.grid)    
            
    def InitializeFields(self):
        """
        Return dictionary of all fields.
        """
 
        fields = {}
        for field in FieldList:
            fields[field] = np.array(eval("map(self.Initialize{0}, self.grid)".format(field)))
        
        if self.OutputRates:
            for i in xrange(3):
                fields['PhotoIonizationRate%i' % i] = np.zeros_like(fields[fields.keys()[0]])
                fields['PhotoHeatingRate%i' % i] = np.zeros_like(fields[fields.keys()[0]])
                fields['CollisionalIonizationRate%i' % i] = np.zeros_like(fields[fields.keys()[0]])
                fields['RadiativeRecombinationRate%i' % i] = np.zeros_like(fields[fields.keys()[0]])
                fields['CollisionalExcitationCoolingRate%i' % i] = np.zeros_like(fields[fields.keys()[0]])
                fields['CollisionalIonzationCoolingRate%i' % i] = np.zeros_like(fields[fields.keys()[0]])
                fields['RecombinationCoolingRate%i' % i] = np.zeros_like(fields[fields.keys()[0]])
                fields['CollisionalExcitationCoolingRate%i' % i] = np.zeros_like(fields[fields.keys()[0]])
                    
                fields['SecondaryIonizationRate%i' % i] = np.zeros([len(fields[fields.keys()[0]]), 3])    
                                
                if i == 2:
                    fields['DielectricRecombinationRate'] = np.zeros_like(fields[fields.keys()[0]])
                    fields['DielectricRecombinationCoolingRate'] = np.zeros_like(fields[fields.keys()[0]])
        
        # Additional fields
        fields['dtPhoton'] = np.ones_like(fields[fields.keys()[0]])
        fields["ODEIterations"] = np.zeros_like(fields[fields.keys()[0]])        
        fields['ODEIterationRate'] = np.zeros_like(fields[fields.keys()[0]])        
        fields['RootFinderIterations'] = np.zeros([len(fields[fields.keys()[0]]), 4])
        fields['OpticalDepth'] = np.zeros([len(fields[fields.keys()[0]]), 3])        
                
        return fields                
                        
    def InitializeDensity(self, cell):
        """
        Initialize the gas density - depends on parameter DensityProfile as follows:
        
            DensityProfile:
                0: Uniform density given by InitialDensity parameter.
                1: Uniform density given by cosmic mean at z = InitialRedshift.
        """        
                
        if self.DensityProfile == 0: 
            density = self.InitialDensity * m_H
        if self.DensityProfile == 1: 
            density = self.Cosmology.MeanBaryonDensity(self.InitialRedshift)
        
        if self.Clump: 
            if self.ClumpDensityProfile == 0:
                if (cell >= (self.ClumpPosition - self.ClumpRadius)) and (cell <= (self.ClumpPosition + self.ClumpRadius)):
                    density *= self.ClumpOverdensity
            if self.ClumpDensityProfile == 1:
                density += density * self.ClumpOverdensity * np.exp(-(cell - self.ClumpPosition)**2 / 2. / self.ClumpRadius**2)
                        
        return density
        
    def InitializeTemperature(self, cell):
        """
        Initialize temperature - depends on parameter TemperatureProfile as follows:
        
            TemperatureProfile:
                0: Uniform temperature given by InitialTemperature
                1: Uniform temperature given assuming the gas decouples from the CMB at z = 250
                2: Gas within StartRadius at 10^4 K, InitialTemperature elsewhere
        """
                
        if self.TemperatureProfile == 0: temperature = self.InitialTemperature    
        if self.TemperatureProfile == 1: temperature = 2.725 * (1. + self.InitialRedshift)**3. / 251.
        if self.TemperatureProfile == 2: temperature = self.InitialTemperature
            
        if self.Clump:
            if (cell >= (self.ClumpPosition - self.ClumpRadius)) and (cell <= (self.ClumpPosition + self.ClumpRadius)):
                temperature = self.ClumpTemperature
        
        return temperature
        
    def InitializeIonization(self, cell):
        """
        Initialize ionization state - depends on parameter IonizationProfile as follows:
        
            IonizationProfile:
                0: Uniform ionization state given by InitialHIIFraction
                1: Gas within 'StartRadius' has x_i = 0.9999, InitialHIIFraction elsewhere
                   
        Returns the HII fraction in 'cell'.
        """
                
        if self.IonizationProfile == 0: 
            ionization = self.InitialHIIFraction 
        
        return ionization    
        
    def InitializeHIDensity(self, cell):
        """
        Initialize neutral hydrogen density.
        """
                
        return (1. - self.Y) * (1. - self.ionization[cell]) * self.density[cell] / m_H
    
    def InitializeHIIDensity(self, cell):
        """
        Initialize ionized hydrogen density.
        """
                
        return (1. - self.Y) * self.ionization[cell] * self.density[cell] / m_H
        
    def InitializeHeIDensity(self, cell):
        """
        Initialize neutral helium density - initial ionized fraction of helium is assumed
        to be the same as that of hydrogen.
        """
                
        return self.Y * (1. - self.ionization[cell]) * self.density[cell] / 4. / m_H
        
    def InitializeHeIIDensity(self, cell):
        """
        Initialize ionized helium density - initial ionized fraction of helium is assumed
        to be the same as that of hydrogen.
        """
        
        return self.Y * self.ionization[cell] * self.density[cell] / 4. / m_H
        
    def InitializeHeIIIDensity(self, cell):
        """
        Initialize doubly ionized helium density - assumed to be very small (can't be exactly zero or it will crash the root finder).
        """
        
        return self.Y * self.MinimumSpeciesFraction * self.density[cell] / 4. / m_H
        
    def InitializeElectronDensity(self, cell):
        """
        Initialize electron density - n_e = n_HII + n_HeII + 2n_HeIII (I'm pretty sure the equation in 
        Thomas and Zaroubi 2007 is wrong - they had n_e = n_HII + n_HeI + 2n_HeII).
        """
        
        return self.InitializeHIIDensity(cell) + self.InitializeHeIIDensity(cell) + 2. * self.InitializeHeIIIDensity(cell)
        
        