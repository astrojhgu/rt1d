"""
RadiationSource.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-08-26.

Description: Radiation source class, contains all functions associated with initializing a 
radiation source.  Mainly used for calculating and normalizing it's luminosity with time.

Notes: 
     
"""

import numpy as np
from Constants import *
from scipy.integrate import quad
import numpy as np

sigma_SB = 2.0 * np.pi**5 * k_B**4 / 15.0 / c**2 / h**3     # Stefan-Boltzmann constant
lsun = 3.839 * 10**33                                       # Solar luminosity - erg / s

SchaererTable = {
                "Mass": [120, 200, 300, 400, 500, 1000], 
                "Temperature": [4.981, 4.999, 5.007, 5.028, 5.029, 5.026],
                "Luminosity": [6.243, 6.574, 6.819, 6.984, 7.106, 7.444]
                }
                
class RadiationSource:
    def __init__(self, pf):
        self.pf = pf
        self.s_type = pf["SourceSpectrum"]
        
        if self.s_type == 0:
            """
            Blackbody.
            """
            self.T = pf["SourceTemperature"]
            self.R = pf["SourceRadius"]
        
        if self.s_type == 1:
            """
            Population III star (Schaerer 2002, Table 3).
            """
            self.T = pf["SourceTemperature"]
            self.M = pf["SourceMass"]
        
        if self.s_type == 2:
            """
            Power-law source, break energy of 1 keV (Madau 2004).
            """
            self.alpha = -pf["SourcePowerLawIndex"]           
       
        # should only need to do this once
        self.LuminosityNormalization = self.NormalizeLuminosity()
        
    def Spectrum(self, E):
        """
        Return the fraction of the bolometric luminosity emitted at this energy.  This quantity is dimensionless.
        """
                
        return self.LuminosityNormalization * self.SpecificIntensity(E) / self.BolometricLuminosity()
                
    def SpecificIntensity(self, E):    
        """ 
        Calculates the specific intensity at energy E for a given spectrum.  To get the bolometric luminosity of the object, 
        one must integrate over energy, multiply by 4 * pi * r^2, and also multiply by the normalization factor.
        
            Units: erg / s / cm^2
            
        """
        
        if self.s_type == 0 or self.s_type == 1:
            """
            Why doesn't this integrate to sigma * T^4 off the bat?
            """
            return 2.0 * (E * erg_per_ev)**3 * (np.exp(E * erg_per_ev / k_B / self.T) - 1.0)**-1 / h**2 / c**2
            
        if self.s_type == 2:
            """
            A simple power law X-ray spectrum with spectral index alpha = -1.0 and break 
            energy h*nu_0 = 1keV (Madau et al. 2004).  Returns the fraction of the total 
            energy emitted at energy E.
            """
            
            return (E / 1000.0)**self.alpha
        
    def NormalizeLuminosity(self):
        """
        Returns a constant that normalizes a given spectrum to its bolometric luminosity.
        """
        
        if self.s_type == 0 or self.s_type == 1:
            integral, err = quad(self.SpecificIntensity, 0, np.inf)
            return self.BolometricLuminosity(0.0) / integral                                    
        
    def BolometricLuminosity(self, t = 0.0):
        """
        Returns the bolometric luminosity of a source in units of erg/s.  For accreting black holes, the 
        bolometric luminosity can increase with time so we'll leave ourselves the opportunity to do this
        in an efficient fashion later on via the 't' argument.
        """
        
        if self.s_type == 0:
            return sigma_SB * self.T**4 * 4.0 * np.pi * (self.R * cm_per_rsun)**2
        
        if self.s_type == 1:
            return 10**SchaererTable["Luminosity"][SchaererTable["Mass"].index(self.M)] * lsun
            
            
            
            
            
            
    
    