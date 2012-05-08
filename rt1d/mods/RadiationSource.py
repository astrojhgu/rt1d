"""
RadiationSource.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-10-26.

Description: Radiation source class, contains all functions associated with initializing a 
radiation source.  Mainly used for calculating and normalizing it's luminosity with time.

Notes: 

SourceType = 0: Monochromatic/Polychromatic emission of SpectrumPhotonLuminosity photons per second.
  -The difference between this and SourceType > 0 is that the bolometric luminosity may change depending
   on your choice of bins.
     
     
"""

import numpy as np
from Integrate import simpson as integrate                  # why did scipy.integrate.quad mess up LphotNorm? 
from scipy.integrate import quad
from .ComputeCrossSections import PhotoIonizationCrossSection as sigma_E

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
y_He = 0.08
xi = (3. / 7.)**0.5 * (6. / 7.)**3                

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
        
        self.tau = pf.SourceLifetime * pf.TimeUnits
                
        # Spectrum bounds - shorthand
        self.Emin = pf.SpectrumMinEnergy
        self.Emax = pf.SpectrumMaxEnergy
        self.EminNorm = pf.SpectrumMinNormEnergy
        self.EmaxNorm = pf.SpectrumMaxNormEnergy

        self.E = np.array(pf.DiscreteSpectrumSED)
        self.F = np.array(pf.DiscreteSpectrumRelLum)      
        
        # SourceType 0, 1, 2
        self.Lph = pf.SpectrumPhotonLuminosity
        
        # SourceType = 1, 2
        self.T = pf.SourceTemperature
        
        # SourceType = 2, 3, 4, 5
        self.M = pf.SourceMass
        
        # SourceType = 3
        self.alpha = -pf.SpectrumPowerLawIndex
        self.epsilon = pf.SourceRadiativeEfficiency
        
        # SourceType = 4
        self.NHI = pf.SpectrumAbsorbingColumn
        
        # SourceType = 5
        self.fdisk = pf.SpectrumDiskFraction
        self.fcol = pf.SpectrumColorCorrection
        self.Rg = G * self.M * g_per_msun / c**2
        
        if not pf.SourceISCO:
            self.r_in = 6. * self.Rg
        else:
            self.r_in = pf.SourceISCO    
            
        self.Lbol = self.BolometricLuminosity(0.0)
        self.T_in = (self.Lbol * xi**2 * self.fcol**4 / 4. * np.pi / sigma_SB / self.r_in**2)**0.25    
             
        # Number of ionizing photons per cm^2 of surface area for BB of temperature self.T.  
        # Use to solve for stellar radius (which we need to get Lbol).  The factor of pi gets rid of the / sr units
        if self.pf.SourceType == 0:
            self.Lbol = self.Lph * self.F * self.E * erg_per_ev 
            self.Qdot = self.F * self.Lph
        elif self.pf.SourceType in [1, 2]:
            self.LphNorm = np.pi * 2. * (k_B * self.T)**3 * \
                integrate(lambda x: x**2 / (np.exp(x) - 1.), 
                13.6 * erg_per_ev / k_B / self.T, big_number, epsrel = 1e-12)[0] / h**3 / c**2 
            self.R = np.sqrt(self.Lph / 4. / np.pi / self.LphNorm)        
            self.Lbol = 4. * np.pi * self.R**2 * sigma_SB * self.T**4
            self.Qdot = self.Lbol * self.F / self.E / erg_per_ev 
             
        # Join MCD and PL spectra
        if self.pf.SourceType == 5:
            self.MCDNormalization, self.PLNormalization = self.NormalizeDisk()
            
        # Normalize spectrum
        self.LuminosityNormalization = self.NormalizeLuminosity()       
                                        
    def Spectrum(self, E, Lbol = None):
        """
        Return the fraction of the bolometric luminosity emitted at this energy.  This quantity is dimensionless, and its integral should be 1.
        """
        
        if self.pf.DiscreteSpectrum == 1: 
            return self.F  
        else: 
            if not Lbol:
                Lbol = self.BolometricLuminosity()                
                        
            return self.LuminosityNormalization * self.SpecificIntensity(E) / Lbol        
                
    def IonizingPhotonLuminosity(self, t = 0, bin = None):
        """
        Return Qdot (photons / s) for this source at energy E.
        """
        
        if self.pf.SourceType in [0, 1, 2]:
            return self.Qdot[bin]
        else:
            return self.BolometricLuminosity(t) * self.F[bin] / self.E[bin] / erg_per_ev          
                
    def SpecificIntensity(self, E):    
        """ 
        Calculates the specific intensity at energy E for a given spectrum.  To get the bolometric luminosity of the object, 
        one must integrate over energy, multiply by 4 * pi * r^2, and also multiply by the normalization factor.
        
            Units: erg / s / cm^2
            
        """       
        if self.pf.SourceType == 0: 
            return self.F[0]
        if self.pf.SourceType in [1, 2]:
            return self.BlackBody(E)
        if self.pf.SourceType == 3: 
            return self.PowerLaw(E)
        if self.pf.SourceType == 4: 
            return self.AbsorbedPowerLaw(E)
        if self.pf.SourceType == 5:
            return self.MCDPL(E)
        
    def BlackBody(self, E, T = None):
        """
        Returns specific intensity of blackbody at self.T.
        """        
        
        if T is None:
            T = self.T
        
        nu = E * erg_per_ev / h
        return 2.0 * h * nu**3 / c**2 / (np.exp(h * nu / k_B / self.T) - 1.0)
        
    def PowerLaw(self, E):    
        """
        A simple power law X-ray spectrum with spectral index alpha and break 
        energy h*nu_0 = 1keV (Madau et al. 2004).  Unlike the previous two spectral types,
        this quantity is completely unnormalized and cannot be considered a specific intensity.
        """
        return E * (E / 1000.0)**self.alpha
        
    def AbsorbedPowerLaw(self, E):
        """
        Same as PowerLaw, except we apply an instrinsic absorbing column as in 
        Kramer & Haiman 2008.
        """    
        
        return self.PowerLaw(E) * np.exp(-self.NHI * (sigma_E(E, 0) + y_He * sigma_E(E, 1)))
    
    def MultiColorDisk(self, E):
        """
        Soft component of accretion disk spectra.
        """         
        
        integrand = lambda T: (T / self.T_in)**(-11. / 3.) * self.BlackBody(E, T) / self.T_in
        integral = quad(integrand, 0., self.T_in)[0]
        
        return 32. * np.pi**2 * self.r_in**2 * integral / 3.
        
    def MCDPL(self, E):
        """
        Multi-color disk + power-law.
        """    
        
        return self.MCDNormalization * self.MultiColorDisk(E) + self.PLNormalization * self.PowerLaw(E)        
        
    def NormalizeLuminosity(self):
        """
        Returns a constant that normalizes a given spectrum to its bolometric luminosity.
        """            

        if self.pf.DiscreteSpectrum == 1:
            integral = 1.0
        
        else:
        
            if self.pf.SourceType in [1, 2]:
                integral = integrate(self.SpecificIntensity, small_number, big_number)[0]
                
            elif self.pf.SourceType in [3, 4]:
                if self.alpha == -1.0: 
                    integral = (1. / 1000.0**self.alpha) * (self.EmaxNorm - self.EminNorm)
                elif self.alpha == -2.0: 
                    integral = (1. / 1000.0**self.alpha) * np.log(self.EmaxNorm / self.EminNorm)    
                else: 
                    integral = (1. / 1000.0**self.alpha) * (1.0 / (self.alpha + 2.0)) * \
                    (self.EmaxNorm**(self.alpha + 2.0) - self.EminNorm**(self.alpha + 2.0))
        
            elif self.pf.SourceType >= 2:
                integral, err = quad(self.SpecificIntensity, self.EminNorm, self.EmaxNorm)             
                     
        
        return self.BolometricLuminosity(0.0) / integral 
    
    def NormalizeDisk(self):
        """
        Join MCD and PL components of MCDPL spectrum.
        """     
        
        mcd_integral, err = quad(self.MultiColorDisk, self.EminNorm, self.EmaxNorm)
        pl_integral, err = quad(self.PowerLaw, self.EminNorm, self.EmaxNorm)
        
        Lbol = self.BolometricLuminosity(0.0)
        
        return self.fdisk * Lbol / mcd_integral, (1. - self.fdisk) * Lbol / pl_integral
        
    def BolometricLuminosity(self, t = 0.0):
        """
        Returns the bolometric luminosity of a source in units of erg/s.  For accreting black holes, the 
        bolometric luminosity will increase with time, hence the optional 't' argument.
        """
        
        if t >= self.tau: 
            return 0.0
        
        if self.pf.SourceType == 0: 
            return self.Lph / (np.sum(self.F / self.E / erg_per_ev))
        
        if self.pf.SourceType == 1:
            return self.Lbol
        
        if self.pf.SourceType == 2:
            return 10**SchaererTable["Luminosity"][SchaererTable["Mass"].index(self.M)] * lsun
            
        if self.pf.SourceType > 2:
            Mnow = self.M * np.exp( ((1.0 - self.epsilon) / self.epsilon) * t / t_edd)
            return self.epsilon * 4.0 * np.pi * G * Mnow * g_per_msun * m_p * c / sigma_T
    
    def SpectrumCDF(self, E):
        """
        Returns cumulative energy output contributed by photons at or less than energy E.
        """    
        
        return integrate(self.Spectrum, small_number, E)[0] 
    
    def SpectrumMedian(self, energies = None):
        """
        Compute median emission energy from spectrum CDF.
        """
        
        if energies is None:
            energies = np.linspace(self.EminNorm, self.EmaxNorm, 200)
        
        if not hasattr('self', 'cdf'):
            cdf = []
            for energy in energies:
                cdf.append(self.SpectrumCDF(energy))
                
            self.cdf = np.array(cdf)
            
        return np.interp(0.5, self.cdf, energies)
    
    def SpectrumMean(self):
        """
        Mean emission energy.
        """        
        
        integrand = lambda E: self.Spectrum(E) * E
        
        return integrate(integrand, self.EminNorm, self.EmaxNorm)[0] 
        
            
