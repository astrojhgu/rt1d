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
from Integrate import simpson as integrate
from scipy.integrate import quad

h = 6.626068e-27     			                            # Planck's constant - [h] = erg*s
k_B = 1.3806503e-16     			                        # Boltzmann's constant - [k_B] = erg/K
c = 29979245800.0 				                            # Speed of light - [c] = cm/s
G = 6.673e-8     				                            # Gravitational constant - [G] = cm^3/g/s^2
sigma_SB = 2.0 * np.pi**5 * k_B**4 / 15.0 / c**2 / h**3     # Stefan-Boltzmann constant - [sigma_SB] = erg / cm^2 / deg^4 / s
lsun = 3.839e33                                             # Solar luminosity - erg / s
m_p = 1.67262158e-24    		                            # Proton mass - [m_p] = g
sigma_T = 6.65e-25	        		                        # Cross section for Thomson scattering - [sigma_T] = cm^2
g_per_msun = 1.98892e33                                     # Mass of the sun - [g_per_sun] = g
erg_per_ev = 1.60217646e-19 / 1e-7                          # Conversion between ergs and eV
cm_per_rsun = 695500.0 * 1e5                                # Radius of the sun - [cm_per_rsun] = cm
s_per_yr = 365.25 * 24 * 3600                               # Seconds per year
t_edd = 0.45 * 1e9 * s_per_yr                               # Eddington timescale (see eq. 1 in Volonteri & Rees 2005) 

np.seterr(all = 'ignore')   # exp overflow occurs when integrating BB - will return 0 as it should for x large

SchaererTable = {
                "Mass": [5, 9, 15, 25, 40, 60, 80, 120, 200, 300, 400, 500, 1000], 
                "Temperature": [4.44, 4.622, 4.759, 4.85, 4.9, 4.943, 4.97, 4.981, 4.999, 5.007, 5.028, 5.029, 5.026],
                "Luminosity": [2.87, 3.709, 4.324, 4.89, 5.42, 5.715, 5.947, 6.243, 6.574, 6.819, 6.984, 7.106, 7.444]
                }

small_number = 1e-3                
big_number = 1e5
                                
class RadiationSource:
    def __init__(self, pf):
        self.pf = pf
        self.SourceType = pf["SourceType"]
        self.DiscreteSpectrum = pf["DiscreteSpectrum"]
        self.tau = pf["SourceLifetime"]
        self.TimeUnits = pf["TimeUnits"]
                
        # Spectrum bounds
        self.Emin = pf["SpectrumMinEnergy"]
        self.Emax = pf["SpectrumMaxEnergy"]
        self.EminNorm = pf["SpectrumMinNormEnergy"]
        self.EmaxNorm = pf["SpectrumMaxNormEnergy"]
                    
        self.E = np.array(pf["DiscreteSpectrumSED"])
        self.F = np.array(pf["DiscreteSpectrumRelLum"])
        self.Lph = pf["SpectrumPhotonLuminosity"]
            
        # SourceType = 1, 2
        self.T = pf["SourceTemperature"]
        
        # Number of ionizing photons per cm^2 of surface area for BB of temperature self.T.  
        # Use to solve for stellar radius (which we need to get Lbol).  The factor of pi gets rid of the / sr units
        self.LphNormIon = np.pi * 2. * (k_B * self.T)**3 * integrate(lambda x: x**2 / (np.exp(x) - 1.), 13.6 * erg_per_ev / k_B / self.T, big_number, tol = 1e-12) / h**3 / c**2 
        
        self.R = np.sqrt(self.Lph / 4. / np.pi / self.LphNormIon)        
        self.Lbol = 4. * np.pi * self.R**2 * sigma_SB * self.T**4
                        
        # SourceType = 2, 3
        self.M = pf["SourceMass"]
        self.FixedSourceMass = pf["FixedSourceMass"]
        
        # SourceType = 3
        self.alpha = -pf["SpectrumPowerLawIndex"] 
        self.epsilon = pf["SourceRadiativeEfficiency"] 
                        
        # Normalize spectrum
        self.LuminosityNormalization = self.NormalizeLuminosity()
        
    def Spectrum(self, E):
        """
        Return the fraction of the bolometric luminosity emitted at this energy.  This quantity is dimensionless, and its integral should be 1.
        """
                        
        if self.DiscreteSpectrum == 1: return self.F  
        else: return self.LuminosityNormalization * self.SpecificIntensity(E) / self.BolometricLuminosity()
                
    def SpecificIntensity(self, E):    
        """ 
        Calculates the specific intensity at energy E for a given spectrum.  To get the bolometric luminosity of the object, 
        one must integrate over energy, multiply by 4 * pi * r^2, and also multiply by the normalization factor.
        
            Units: erg / s / cm^2
            
        """       
        if self.SourceType == 0: return 1.0
        if self.SourceType == 1 or self.SourceType == 2: return self.BlackBody(E)
        if self.SourceType == 3: return self.PowerLaw(E)
        
    def BlackBody(self, E):
        """
        Returns specific intensity of blackbody at self.T.
        """        
        nu = E * erg_per_ev / h
        return 2.0 * h * nu**3 / c**2 / (np.exp(h * nu / k_B / self.T) - 1.0)
        
    def PowerLaw(self, E):    
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

        if self.DiscreteSpectrum == 1:
            integral = 1.0
        
        else:
        
            if self.SourceType == 1 or self.SourceType == 2:
                integral = integrate(self.SpecificIntensity, small_number, big_number)
                
            if self.SourceType == 3:
                if self.alpha == -1.0: 
                    integral = (1. / 1000.0**self.alpha) * (self.EmaxNorm - self.EminNorm)
                elif self.alpha == -2.0: 
                    integral = (1. / 1000.0**self.alpha) * np.log(self.EmaxNorm / self.EminNorm)    
                else: 
                    integral = (1. / 1000.0**self.alpha) * (1.0 / (self.alpha + 2.0)) * \
                    (self.EmaxNorm**(self.alpha + 2.0) - self.EminNorm**(self.alpha + 2.0))  
        
        return self.BolometricLuminosity(0.0) / integral  
        
    def BolometricLuminosity(self, t = 0.0):
        """
        Returns the bolometric luminosity of a source in units of erg/s.  For accreting black holes, the 
        bolometric luminosity will increase with time, hence the optional 't' argument.
        """
        
        if (t / self.TimeUnits) > self.tau: return 0.0
        
        if self.SourceType == 0: 
            return self.Lph / (np.sum(self.F / self.E / erg_per_ev))
        
        if self.SourceType == 1:
            return self.Lbol
        
        if self.SourceType == 2:
            return 10**SchaererTable["Luminosity"][SchaererTable["Mass"].index(self.M)] * lsun
            
        if self.SourceType > 2:
            if self.FixedSourceMass: Mnow = self.M
            else: Mnow = self.M * np.exp( ((1.0 - self.epsilon) / self.epsilon) * t / t_edd)
            return self.epsilon * 4.0 * np.pi * G * Mnow * g_per_msun * m_p * c / sigma_T
            
    def SpectrumCDF(self, E):
        """
        Returns cumulative energy output contributed by photons at or less than energy E.
        """    
        
        return integrate(self.Spectrum, small_number, E)      
            
