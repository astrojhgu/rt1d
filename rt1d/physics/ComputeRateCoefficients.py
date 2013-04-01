"""

RateCoefficients.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Dec 26 20:59:24 2012

Description: Rate coefficients for hydrogen and helium.  Currently using
Fukugita & Kawasaki (1994). Would be nice to include rates from other sources.

"""

import numpy as np

class RateCoefficients:
    def __init__(self, grid, source='fk94'):
        self.grid = grid
        self.cosm = grid.cosm   
        
    @property
    def RecombinationMethod(self):    
        if self.pf is not None:
            return self.pf['recombination']
        else:
            return 'B'          
        
    def CollisionalIonizationRate(self, species, T):
        """
        Collisional ionization rate which we denote elsewhere as Beta.
        """    
        
        if species == 0:  
            return 5.85e-11 * np.sqrt(T) * (1. + np.sqrt(T / 1.e5))**-1. * np.exp(-1.578e5 / T)    
          
        if species == 1:    
            return 2.38e-11 * np.sqrt(T) * (1. + np.sqrt(T / 1.e5))**-1. * np.exp(-2.853e5 / T) 
        
        if species == 2:
            return 5.68e-12 * np.sqrt(T) * (1. + np.sqrt(T / 1.e5))**-1. * np.exp(-6.315e5 / T)     
        
    def RadiativeRecombinationRate(self, species, T):
        """
        Coefficient for radiative recombination.  Here, species = 0, 1, 2
        refers to HII, HeII, and HeIII.
        """
        
        if self.grid.recombination_method == 'A':
            if species == 0:
                return 6.28e-11 * T**-0.5 * (T / 1e3)**-0.2 * (1. + (T / 1e6)**0.7)**-1.
            elif species == 1:
                return 1.5e-10 * T**-0.6353
            elif species == 2:
                return 3.36e-10 * T**-0.5 * (T / 1e3)**-0.2 * (1. + (T / 4e6)**0.7)**-1.
        elif self.grid.recombination_method == 'B':
            if species == 0:
                return 2.6e-13 * (T / 1.e4)**-0.85 
            elif species == 1:
                return 9.94e-11 * T**-0.6687
            elif species == 2:
                alpha = 3.36e-10 * T**-0.5 * (T / 1e3)**-0.2 * (1. + (T / 4.e6)**0.7)**-1 # To n >= 1
                alpha[T < 2.2e4] *= (1.11 - 0.044 * np.log(T[T < 2.2e4]))   # To n >= 2                   
                alpha[T >= 2.2e4] *= (1.43 - 0.076 * np.log(T[T >= 2.2e4])) # To n >= 2
                
                return alpha
        else:
            print 'Unrecognized RecombinationMethod.  Should be A or B.'
            return 0.0          
        
    def DielectricRecombinationRate(self, T):
        """
        Dielectric recombination coefficient for Helium.
        """
        
        return 1.9e-3 * T**-1.5 * np.exp(-4.7e5 / T) * (1. + 0.3 * np.exp(-9.4e4 / T))
        
    def CollisionalIonizationCoolingRate(self, species, T):
        """
        Returns coefficient for cooling by collisional ionization.  These are equations B4.1a, b, and d respectively
        from FK96.
        
            units: erg cm^3 / s
        """
        
        if species == 0: 
            return 1.27e-21 * np.sqrt(T) * (1. + np.sqrt(T / 1e5))**-1. * np.exp(-1.58e5 / T)
        if species == 1: 
            return 9.38e-22 * np.sqrt(T) * (1. + np.sqrt(T / 1e5))**-1. * np.exp(-2.85e5 / T)
        if species == 2: 
            return 4.95e-22 * np.sqrt(T) * (1. + np.sqrt(T / 1e5))**-1. * np.exp(-6.31e5 / T)
            
    def CollisionalExcitationCoolingRate(self, species, T):
        """
        Returns coefficient for cooling by collisional excitation.  These are equations B4.3a, b, and c respectively
        from FK96.
        
            units: erg cm^3 / s
        """
        
        if species == 0: 
            return 7.5e-19 * (1. + np.sqrt(T / 1e5))**-1. * np.exp(-1.18e5 / T)
        if species == 1: 
            return 9.1e-27 * T**-0.1687 * (1. + np.sqrt(T / 1e5))**-1. * np.exp(-1.31e4 / T)   # CONFUSION
        if species == 2: 
            return 5.54e-17 * T**-0.397 * (1. + np.sqrt(T / 1e5))**-1. * np.exp(-4.73e5 / T)    
        
    def RecombinationCoolingRate(self, species, T):
        """
        Returns coefficient for cooling by recombination.  These are equations B4.2a, b, and d respectively
        from FK96.
        
            units: erg cm^3 / s
        """
        
        if species == 0: 
            return 6.5e-27 * np.sqrt(T) * (T / 1e3)**-0.2 * (1.0 + (T / 1e6)**0.7)**-1.0
        if species == 1: 
            return 1.55e-26 * T**0.3647
        if species == 2: 
            return 3.48e-26 * np.sqrt(T) * (T / 1e3)**-0.2 * (1. + (T / 4e6)**0.7)**-1.
        
    def DielectricRecombinationCoolingRate(self, T):
        """
        Returns coefficient for cooling by dielectric recombination.  This is equation B4.2c from FK96.
        
            units: erg cm^3 / s
        """
        
        return 1.24e-13 * T**-1.5 * np.exp(-4.7e5 / T) * (1. + 0.3 * np.exp(-9.4e4 / T))
        
    def HubbleCoolingRate(self, z):
        """
        Just the Hubble parameter. 
        """
        
        return self.cosm.HubbleParameter(z)
        
    def ComptonHeatingRate(self, z, nH, nHe, ne, xHII, Tk):
        """
        Compton heating rate due to electron-CMB scattering.
        """    
        
        if z > self.cosm.zdec:
            Tcmb = self.cosm.TCMB(z)
            ucmb = sigma_SB * Tcmb**4. / 4. / np.pi / c
            
            return 4. * sigma_T * ne * ucmb * k_B * (Tcmb - Tk) / m_e / c
            #return xHII * k_B * (nH + nHe + ne) * 4. * sigma_T * ucmb * \
            #    (Tcmb - Tk) / (1. + self.cosm.y + xHII) / m_e / c   
        else:
            return 0.0
            # Implement source-dependent compton heating 