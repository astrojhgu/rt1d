"""
InitializeGrid.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-10-18.

Description: Construct 1D arrays for our data in code units.  Return as dictionary.

Notes: Non-constant profiles need fixing.
     
"""

import numpy as np
from .Constants import *
from .Cosmology import Cosmology

FieldList = \
    ["Temperature", "HIDensity", "HIIDensity", "HeIDensity", \
     "HeIIDensity", "HeIIIDensity", "ElectronDensity"]        

tiny_number = 1e-12

class InitializeGrid:
    def __init__(self, pf):
        self.pf = pf
        self.cosm = Cosmology(pf)
        
        # Deal with log-grid, compute dx
        self.R0 = pf['StartRadius'] * pf['LengthUnits']
        if pf['LogarithmicGrid']:
            self.r = np.logspace(np.log10(self.R0), \
                np.log10(pf['LengthUnits']), int(pf['GridDimensions']) + 1)
        else:
            self.r = np.linspace(self.R0, pf['LengthUnits'], int(pf['GridDimensions']) + 1)
        
        self.dx = np.diff(self.r)   
        self.r = self.r[0:-1]             
        self.grid = np.arange(len(self.r))
                            
        # Generic data array                
        self.density = np.array(map(self.InitializeDensity, self.grid))
        self.ionization = np.array(map(self.InitializeIonization, self.grid))
                    
    def InitializeFields(self):
        """
        Return dictionary of all fields.
        """
 
        fields = {}
        for field in FieldList:
            fields[field] = np.array(eval("map(self.Initialize{0}, self.grid)".format(field)))
        
        if self.pf['OutputRates']:
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
        fields['Radius'] = self.r
        fields['ShellThickness'] = self.dx  
        fields['PhotonPackages'] = np.zeros(3)
                
        return fields                
                        
    def InitializeDensity(self, cell):
        """
        Initialize the gas density - depends on parameter DensityProfile as follows:
        
            DensityProfile:
                0: Uniform density given by InitialDensity parameter.
                1: Uniform density given by cosmic mean at z = InitialRedshift.
        """        
                
        if self.pf['DensityProfile'] == 0: 
            density = self.pf['DensityUnits']
        if self.pf['DensityProfile'] == 1: 
            density = self.cosm.MeanBaryonDensity(self.pf['InitialRedshift'])
        
        if self.pf['Clump']: 
            if self.pf['ClumpDensityProfile'] == 0:
                if (cell >= self.pf['GridDimensions'] * (self.pf['ClumpPosition'] - self.pf['ClumpRadius'])) and \
                   (cell <= self.pf['GridDimensions'] * (self.pf['ClumpPosition'] + self.pf['ClumpRadius'])):
                    density *= self.pf['ClumpOverdensity']
            if self.pf['ClumpDensityProfile'] == 1:
                density += density * self.pf['ClumpOverdensity'] * \
                    np.exp(-(cell - self.pf['ClumpPosition'] * pf['GridDimensions'])**2 / 2. / self.pf['ClumpRadius']**2)
                        
        return density
        
    def InitializeTemperature(self, cell):
        """
        Initialize temperature - depends on parameter TemperatureProfile as follows:
        
            TemperatureProfile:
                0: Uniform temperature given by InitialTemperature
                1: Uniform temperature assuming Tk = Tcmb before decoupling, and
                   Tk ~ (1 + z)^2 after decoupling.
        """
                
        if self.pf['TemperatureProfile'] == 0: 
            temperature = self.pf['InitialTemperature']    
        if self.pf['TemperatureProfile'] == 1: 
            temperature = self.cosm.Tgas(self.pf['InitialRedshift'])
                
        if self.pf['Clump']:
            if (cell >= self.pf['GridDimensions'] * (self.pf['ClumpPosition'] - self.pf['ClumpRadius'])) and \
               (cell <= self.pf['GridDimensions'] * (self.pf['ClumpPosition'] + self.pf['ClumpRadius'])):
                temperature = self.pf['ClumpTemperature']
        
        return temperature
        
    def InitializeIonization(self, cell):
        """
        Initialize ionization state - depends on parameter IonizationProfile as follows:
        
            IonizationProfile:
                0: Uniform ionization state given by InitialHIIFraction
                1: Gas within 'StartRadius' has x_i = 0.9999, InitialHIIFraction elsewhere
                   
        Returns the HII fraction in 'cell'.
        """
                
        if self.pf['IonizationProfile'] == 0: 
            ionization = self.pf['InitialHIIFraction']
        
        return ionization    
        
    def InitializeHIDensity(self, cell):
        """
        Initialize neutral hydrogen density.
        """
                
        return (1. - self.cosm.Y) * (1. - self.ionization[cell]) * self.density[cell] / m_H
    
    def InitializeHIIDensity(self, cell):
        """
        Initialize ionized hydrogen density.
        """
                
        return (1. - self.cosm.Y) * self.ionization[cell] * self.density[cell] / m_H
        
    def InitializeHeIDensity(self, cell):
        """
        Initialize neutral helium density - initial ionized fraction of helium is assumed
        to be the same as that of hydrogen.
        """
                
        return self.cosm.Y * (1. - self.ionization[cell]) * self.density[cell] / 4. / m_H
        
    def InitializeHeIIDensity(self, cell):
        """
        Initialize ionized helium density - initial ionized fraction of helium is assumed
        to be the same as that of hydrogen.
        """
        
        return self.cosm.Y * self.ionization[cell] * self.density[cell] / 4. / m_H
        
    def InitializeHeIIIDensity(self, cell):
        """
        Initialize doubly ionized helium density - assumed to be very small (can't be exactly zero or it will crash the root finder).
        """
        
        return self.cosm.Y * self.pf.MinimumSpeciesFraction * self.density[cell] / 4. / m_H
        
    def InitializeElectronDensity(self, cell):
        """
        Initialize electron density - n_e = n_HII + n_HeII + 2n_HeIII (I'm pretty sure the equation in 
        Thomas and Zaroubi 2007 is wrong - they had n_e = n_HII + n_HeI + 2n_HeII).
        """
        
        return self.InitializeHIIDensity(cell) + self.InitializeHeIIDensity(cell) + \
            2. * self.InitializeHeIIIDensity(cell)
        
        