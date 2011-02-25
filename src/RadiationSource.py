"""
RadiationSource.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-10-26.

Description: Radiation source class, contains all functions associated with initializing a 
radiation source.  Mainly used for calculating and normalizing it's luminosity with time.

Notes: 
     
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve

h = 6.626068 * 10**-27 			                            # Planck's constant - [h] = erg*s
k_B = 1.3806503 * 10**-16			                        # Boltzmann's constant - [k_B] = erg/K
c = 29979245800.0 				                            # Speed of light - [c] = cm/s
G = 6.673*10**-8 				                            # Gravitational constant - [G] = cm^3/g/s^2
sigma_SB = 2.0 * np.pi**5 * k_B**4 / 15.0 / c**2 / h**3     # Stefan-Boltzmann constant
lsun = 3.839 * 10**33                                       # Solar luminosity - erg / s
m_p = 1.67262158*10**-24		                            # Proton mass - [m_p] = g
sigma_T = 6.65 * 10**-25			                        # Cross section for Thomson scattering - [sigma_T] = cm^2
g_per_msun = 1.98892 * 10**33                               # Mass of the sun - [g_per_sun] = g
erg_per_ev = 1.60217646*10**-19 / 1e-7                      # Conversion between ergs and eV
cm_per_rsun = 695500.0 * 1e5                                # Radius of the sun - [cm_per_rsun] = cm
s_per_yr = 365.25 * 24 * 3600                               # Seconds per year
t_edd = 0.45 * 1e9 * s_per_yr                               # Eddington timescale (see eq. 1 in Volonteri & Rees 2005) 

SchaererTable = {
                "Mass": [5, 9, 15, 25, 40, 60, 80, 120, 200, 300, 400, 500, 1000], 
                "Temperature": [4.44, 4.622, 4.759, 4.85, 4.9, 4.943, 4.97, 4.981, 4.999, 5.007, 5.028, 5.029, 5.026],
                "Luminosity": [2.87, 3.709, 4.324, 4.89, 5.42, 5.715, 5.947, 6.243, 6.574, 6.819, 6.984, 7.106, 7.444]
                }
                
class RadiationSource:
    def __init__(self, pf):
        self.pf = pf
        self.SourceType = pf["SourceType"]
        self.tau = pf["SourceLifetime"]
        self.TimeUnits = pf["TimeUnits"]
        
        # Source discretization
        self.DiscreteSpectrumMethod = pf["DiscreteSpectrumMethod"]
        self.DiscreteSpectrumMinEnergy = pf["DiscreteSpectrumMinEnergy"]
        self.DiscreteSpectrumMaxEnergy = pf["DiscreteSpectrumMaxEnergy"]
        self.DiscreteSpectrumNumberOfBins = pf["DiscreteSpectrumNumberOfBins"]
        self.DiscreteSpectrumBinEdges = pf["DiscreteSpectrumBinEdges"]
        self.DiscreteSpectrumRelLum = pf["DiscreteSpectrumRelLum"] 
        
        # Spectrum bounds
        self.Emin = pf["SpectrumMinEnergy"]
        self.Emax = pf["SpectrumMaxEnergy"]
        self.EminNorm = pf["SpectrumMinNormEnergy"]
        self.EmaxNorm = pf["SpectrumMaxNormEnergy"]
        
        # Set source-specific parameters
        if self.SourceType == 0:
            """
            Source of fixed monochromatic photon flux.
            """
            self.E = pf["DiscreteSpectrumSED"][0]
            self.L = pf["SpectrumPhotonLuminosity"]
        
        if self.SourceType == 1:
            """
            Blackbody.
            """
            self.T = pf["SourceTemperature"]
            self.R = pf["SourceRadius"]
        
        if self.SourceType == 2:
            """
            Population III star (Schaerer 2002, Table 3).
            """
            self.T = pf["SourceTemperature"]
            self.M = pf["SourceMass"]
        
        if self.SourceType == 3:
            """
            Power-law source, break energy of 1 keV (Madau 2004).
            """
            self.M = pf["SourceMass"]
            self.alpha = -pf["SpectrumPowerLawIndex"] 
            self.epsilon = pf["SourceRadiativeEfficiency"] 
                
        # Source discretization
        if self.DiscreteSpectrumMethod == 1:
            self.DiscreteSpectrumSED = np.array(pf["DiscreteSpectrumSED"])
        elif self.DiscreteSpectrumMethod == 2:
            self.DiscreteSpectrumSED = np.linspace(self.DiscreteSpectrumMinEnergy, self.DiscreteSpectrumMaxEnergy, self.DiscreteSpectrumNumberOfBins)
        elif self.DiscreteSpectrumMethod == 3:
            self.DiscreteSpectrumSED = np.logspace(np.log10(self.DiscreteSpectrumMinEnergy), np.log10(self.DiscreteSpectrumMaxEnergy), self.DiscreteSpectrumNumberOfBins)
        elif self.DiscreteSpectrumMethod >= 4:
            
            self.DiscreteSpectrumSED = np.zeros_like(self.DiscreteSpectrumBinEdges)
            for i, edge in enumerate(self.DiscreteSpectrumBinEdges): 
                
                # Calculate bandpass upper limit
                if i < len(self.DiscreteSpectrumBinEdges) - 1: ulim = self.DiscreteSpectrumBinEdges[i + 1]
                else: ulim = self.DiscreteSpectrumMaxEnergy
                
                E_exp = lambda E: E * self.SpecificIntensity(E)  # Mean energy
                E_med = lambda E: (quad(self.SpecificIntensity, edge, E)[0] / quad(self.SpecificIntensity, edge, ulim)[0]) - 0.5 # Median energy
                
                if self.DiscreteSpectrumMethod == 4:
                    self.DiscreteSpectrumSED[i] = quad(E_exp, edge, ulim)[0] / quad(self.SpecificIntensity, edge, ulim)[0]
                elif self.DiscreteSpectrumMethod == 5:
                    self.DiscreteSpectrumSED[i] = fsolve(E_med, np.mean([edge, ulim]))
                    
                # Force all energy above last bin to be emitted at last bin    
                self.DiscreteSpectrumSED[-1] = self.DiscreteSpectrumBinEdges[-1]
                self.DiscreteSpectrumNumberOfBins = len(self.DiscreteSpectrumSED)
        else:
            pass
            
        # Normalize spectrum
        self.LuminosityNormalization = self.NormalizeLuminosity()
        
    def Spectrum(self, E):
        """
        Return the fraction of the bolometric luminosity emitted at this energy.  This quantity is dimensionless.
        """
                        
        if self.DiscreteSpectrumMethod == 1:
            return self.DiscreteSpectrumRelLum    
        else:                
            return self.LuminosityNormalization * self.SpecificIntensity(E) / self.BolometricLuminosity()
                
    def SpecificIntensity(self, E):    
        """ 
        Calculates the specific intensity at energy E for a given spectrum.  To get the bolometric luminosity of the object, 
        one must integrate over energy, multiply by 4 * pi * r^2, and also multiply by the normalization factor.
        
            Units: erg / s / cm^2
            
        """       
        
        if self.DiscreteSpectrumMethod < 4:
            if self.SourceType == 0:
                return 1.0
            
            if self.SourceType == 1 or self.SourceType == 2:
                return self.BlackBody(E)
                
            if self.SourceType == 3:
                return self.PowerLaw(E)
        
        
        # Bandpass averaging    
        else:
            """
            find bin edges for this energy.
            integrate spectrum over this energy range
            return integrated intensity over bandpass
            
            """
                        
            for i, edge in enumerate(self.DiscreteSpectrumBinEdges):
                if (E < self.DiscreteSpectrumBinEdges[i]):
                    return 0.0
                    
                elif i < len(self.DiscreteSpectrumBinEdges) - 1:
                    
                    if (E > edge) and (E < self.DiscreteSpectrumBinEdges[i + 1]):
                        
                        if self.SourceType == 1 or self.SourceType == 2:
                            I = quad(self.BlackBody, edge, self.DiscreteSpectrumBinEdges[i + 1])[0]
                        elif self.SourceType == 3:   
                            I = quad(self.PowerLaw, edge, self.DiscreteSpectrumBinEdges[i + 1])[0]

                        return I
                
                else:
                    if E >= self.DiscreteSpectrumBinEdges[-1]:
                        if self.SourceType == 1 or self.SourceType == 2:
                            I = quad(self.BlackBody, edge, np.inf)[0]
                        elif self.SourceType == 3:   
                            I = quad(self.PowerLaw, edge, self.Emax)[0]

                        return I
        
    def BlackBody(self, E):
        """
        Returns specific intensity of blackbody at self.T.  Not yet normalized.
        """        
        return 2.0 * (E * erg_per_ev)**3 * (np.exp(E * erg_per_ev / k_B / self.T) - 1.0)**-1 / h**2 / c**2
                        
    def PowerLaw(self, E):
        """
        Returns specific intensity of power law spectrum with index self.alpha.  Should generalize
        so that we can adjust the break energy (currently set to 1 keV) in a parameter.
        """    
        """
        A simple power law X-ray spectrum with spectral index alpha and break 
        energy h*nu_0 = 1keV (Madau et al. 2004).  Unlike the previous two spectral types,
        this quantity is completely unnormalized and cannot be considered a specific intensity.
        """
        
        return E * (E / 1000.0)**self.alpha
        
        
    def NormalizeLuminosity(self):
        """
        Returns a constant that normalizes a given spectrum to its bolometric luminosity.
        """
            
        if self.DiscreteSpectrumMethod == 0:
            if self.SourceType == 0:
                integral = 1.0
            
            if self.SourceType == 1 or self.SourceType == 2:
                integral, err = quad(self.SpecificIntensity, 0, np.inf)
                
            if self.SourceType == 3:
                if self.alpha == -1.0: 
                    integral = (1. / 1000.0**self.alpha) * (self.EmaxNorm - self.EminNorm)
                elif self.alpha == -2.0: 
                    integral = (1. / 1000.0**self.alpha) * np.log(self.EmaxNorm / self.EminNorm)    
                else: 
                    integral = (1. / 1000.0**self.alpha) * (1.0 / (self.alpha + 2.0)) * \
                    (self.EmaxNorm**(self.alpha + 2.0) - self.EminNorm**(self.alpha + 2.0))  
                    
        elif self.DiscreteSpectrumMethod < 4:
            integral = np.sum(self.SpecificIntensity(self.DiscreteSpectrumSED)) 
            
        else:
            integral = 0
            for i, element in enumerate(self.DiscreteSpectrumSED):
                integral += self.SpecificIntensity(element)  
                                                                                                                                                        
        return self.BolometricLuminosity(0.0) / integral  
        
    def BolometricLuminosity(self, t = 0.0):
        """
        Returns the bolometric luminosity of a source in units of erg/s.  For accreting black holes, the 
        bolometric luminosity can increase with time, hence the optional 't' argument.
        """
        
        if (t / self.TimeUnits) > self.tau: return 0.0
        
        if self.SourceType == 0:
            return self.L * self.E * erg_per_ev
        
        if self.SourceType == 1:
            return sigma_SB * self.T**4 * 4.0 * np.pi * (self.R * cm_per_rsun)**2
        
        if self.SourceType == 2:
            return 10**SchaererTable["Luminosity"][SchaererTable["Mass"].index(self.M)] * lsun
            
        if self.SourceType > 2:
            Mnow = self.M * np.exp( ((1.0 - self.epsilon) / self.epsilon) * t / t_edd)
            return self.epsilon * 4.0 * np.pi * G * Mnow * g_per_msun * m_p * c / sigma_T
            
            
            
            
            
            
    
    