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
from scipy.integrate import quad
from Integrate import simpson as integrate                  # why did scipy.integrate.quad mess up LphotNorm? 
from .Constants import *
from .ReadParameterFile import dotdictify
from .ComputeCrossSections import PhotoIonizationCrossSection as sigma_E

np.seterr(all = 'ignore')   # exp overflow occurs when integrating BB - will return 0 as it should for x large

SchaererTable = {
                "Mass": [5, 9, 15, 25, 40, 60, 80, 120, 200, 300, 400, 500, 1000], 
                "Temperature": [4.44, 4.622, 4.759, 4.85, 4.9, 4.943, 4.97, 4.981, 4.999, 5.007, 5.028, 5.029, 5.026],
                "Luminosity": [2.87, 3.709, 4.324, 4.89, 5.42, 5.715, 5.947, 6.243, 6.574, 6.819, 6.984, 7.106, 7.444]
                }

small_number = 1e-3                
big_number = 1e5
ls = ['-', '--', ':', '-.']

"""
SourceType = 0  (monochromatic)     Just need DiscreteSpectrum and PhotonLuminosity
SourceType = 1  (star)              Need temperature, PhotonLuminosity
SourceType = 2  (popIII star)       Need mass
SourceType = 3  (BH)                Need Mass, epsilon

SpectrumType = 0 (monochromatic)    
SpectrumType = 1 (blackbody)                    
SpectrumType = 2 (blackbody, but temperature and luminosity from Schaerer)
SpectrumType = 3 (multi-color disk)
SpectrumType = 4 (simple power-law)
"""
                                
class RadiationSource:
    def __init__(self, pf):
        self.pf = pf
        
        self.SpectrumPars = dotdictify(listify(pf))
        self.N = len(self.SpectrumPars.Type)
        
        # Cast types to int to avoid indexing complaints
        self.SpectrumPars.Type = map(int, self.SpectrumPars.Type)
        
        self.Emin = min(self.SpectrumPars.MinEnergy)
        self.Emax = min(self.SpectrumPars.MaxEnergy)   
        self.EminNorm = min(self.SpectrumPars.MinNormEnergy)
        self.EmaxNorm = min(self.SpectrumPars.MaxNormEnergy)
        
        if self.N == 1:
            self.Type = self.SpectrumPars.Type[0]
                    
        # Correct later if using multi-group approach
        self.E = np.array(pf.DiscreteSpectrumSED)
        self.F = np.array(pf.DiscreteSpectrumRelLum)      
        
        self.last_renormalized = 0
        self.tau = pf.SourceLifetime * pf.TimeUnits
        
        # SourceType 0, 1, 2
        self.Lph = pf.SpectrumPhotonLuminosity
        
        # SourceType = 1, 2
        self.T = pf.SourceTemperature
        
        # SourceType >= 3
        self.M = pf.SourceMass 
        self.M0 = pf.SourceMass
        self.epsilon = pf.SourceRadiativeEfficiency
        
        if 3 in self.SpectrumPars.Type:
            self.r_in = self.DiskInnermostRadius(self.M0)
            self.r_out = pf.SourceDiskMaxRadius * self.GravitationalRadius(self.M0)
            self.fcol = self.SpectrumPars.ColorCorrectionFactor[self.SpectrumPars.Type.index(3)]
            self.T_in = self.DiskInnermostTemperature(self.M0)
            self.T_out = self.DiskTemperature(self.M0, self.r_out)

        ### PUT THIS STUFF ELSEWHERE
                                 
        # Number of ionizing photons per cm^2 of surface area for BB of temperature self.T.  
        # Use to solve for stellar radius (which we need to get Lbol).  The factor of pi gets rid of the / sr units
        if pf.SourceType == 0:
            self.Lbol = self.Lph * self.F * self.E * erg_per_ev 
            self.Qdot = self.F * self.Lph
        elif pf.SourceType in [1, 2]:
            self.LphNorm = np.pi * 2. * (k_B * self.T)**3 * \
                integrate(lambda x: x**2 / (np.exp(x) - 1.), 
                13.6 * erg_per_ev / k_B / self.T, big_number, epsrel = 1e-12)[0] / h**3 / c**2 
            self.R = np.sqrt(self.Lph / 4. / np.pi / self.LphNorm)        
            self.Lbol = 4. * np.pi * self.R**2 * sigma_SB * self.T**4
            self.Qdot = self.Lbol * self.F / self.E / erg_per_ev
        else:
            self.Lbol = self.BolometricLuminosity(0.0)    
                
        # Parameters for average AGN spectrum of SOS04.
        self.Alpha = 0.24
        self.Beta = 1.60
        self.Gamma = 1.06
        self.E_1 = 83.
        self.K = 0.0041
        self.E_0 = (self.Beta - self.Alpha) * self.E_1
        self.A = np.exp(2.0 / self.E_1) * 2.0**self.Alpha
        self.B = ((self.E_0**(self.Beta - self.Alpha)) * np.exp(-(self.Beta - self.Alpha))) / \
            (1.0 + (self.K * self.E_0**(self.Beta - self.Gamma)))
            
        # Normalization constants to make the SOS04 spectrum continuous.
        self.SX_Normalization = 1.0
        self.UV_Normalization = self.SX_Normalization * ((self.A * 2000.0**-self.Alpha) * \
            np.exp(-2000.0 / self.E_1)) / ((1.2 * 2000**-1.7) * np.exp(2000.0 / 2000.0))
        self.IR_Normalization = self.UV_Normalization * ((1.2 * 10**-1.7) * np.exp(10.0 / 2000.0)) / \
            (1.2 * 159 * 10**-0.6)
        self.HX_Normalization = self.SX_Normalization * (self.A * self.E_0**-self.Alpha * \
            np.exp(-self.E_0 / self.E_1)) / (self.A * self.B * (1.0 + self.K * self.E_0**(self.Beta - self.Gamma)) * \
            self.E_0**-self.Beta)               
             
        # Normalize spectrum
        self.LuminosityNormalizations = self.NormalizeSpectrumComponents(0.0)
        
        # Multi-group treatment
        Qnorm = pf.SpectrumPhotonLuminosity / self.Lbol / \
            quad(lambda x: self.Spectrum(x) / x, self.EminNorm, self.EmaxNorm)[0]
        if pf.FrequencyAveragedCrossSections:
            self.E = np.zeros(pf.FrequencyGroups)
            self.F = np.ones_like(self.E)
            self.Qdot = np.zeros_like(self.E)
            self.bands = pf.FrequencyBands
            
            if len(self.bands) == (len(self.E) - 1):
                self.bands.append(self.Emax)
            
            for i in xrange(pf.FrequencyGroups):
                L = quad(lambda x: self.Spectrum(x), self.bands[i], self.bands[i + 1])[0]
                Q = quad(lambda x: self.Spectrum(x) / x, self.bands[i], self.bands[i + 1])[0]
                
                self.E[i] = L / Q
                self.Qdot[i] = Qnorm * self.Lbol * Q
                
        # Time evolution
        if pf.SourceTimeEvolution:
            self.logMarr = np.logspace(np.log10(pf.SourceMass), np.log10(pf.SourceFinalMass), pf.MassBins)
            self.Marr = 10**self.logMarr     
                                              
    def GravitationalRadius(self, M):
        """
        Half the Schwartzchild radius.
        """
        return G * M * g_per_msun / c**2    
        
    def SchwartzchildRadius(self, M):
        return 2. * self.GravitationalRadius(M)    
        
    def MassAccretionRate(self, M = None):
        return self.BolometricLuminosity(0, M = M) / self.epsilon / c**2    
        
    def DiskInnermostRadius(self, M):      
        """
        Inner radius of disk.  Unless SourceISCO > 0, will be set to the 
        inner-most stable circular orbit for a BH of mass M.
        """
        if not self.pf.SourceISCO:
            return 6. * self.GravitationalRadius(M)
        else:
            return self.pf.SourceISCO     
            
    def DiskInnermostTemperature(self, M):
        """
        Temperature (in Kelvin) at inner edge of the disk.
        """
        return (3. * G * M * g_per_msun * self.MassAccretionRate(M) / \
            8. / np.pi / self.DiskInnermostRadius(M)**3 / sigma_SB)**0.25
    
    def DiskTemperature(self, M, r):
        return ((3. * G * M * g_per_msun * self.MassAccretionRate(M) / \
            8. / np.pi / r**3 / sigma_SB) * \
            (1. - (self.DiskInnermostRadius(M) / r)**0.5))**0.25
            
    def BlackHoleMass(self, t):
        """
        Compute black hole mass after t (seconds) have elapsed.  Relies on 
        initial mass self.M, and (constant) radiaitive efficiency self.epsilon.
        """        
        
        return self.M0 * np.exp(((1.0 - self.epsilon) / self.epsilon) * t / t_edd)         
    
    def BlackHoleAge(self, M):
        """
        Compute age of black hole based on current time, current mass, and initial mass.
        """            
        
        return np.log(M / self.SourceMass) (self.epsilon / (1. - self.epsilon)) * t_edd
                
    def IonizingPhotonLuminosity(self, t = 0, bin = None):
        """
        Return Qdot (photons / s) for this source at energy E.
        """
        
        if self.pf.SourceType in [0, 1, 2]:
            return self.Qdot[bin]
        else:
            return self.BolometricLuminosity(t) * self.F[bin] / self.E[bin] / erg_per_ev          
              
    def Intensity(self, E, i, Type, t):
        """
        Return quantity *proportional* to fraction of bolometric luminosity emitted
        at photon energy E.  Normalization handled separately.
        """
        
        if Type == 0:
            Lnu = self.F[0]
        elif Type in [1, 2]:
            Lnu = self.BlackBody(E)
        elif Type == 3:
            Lnu = self.MultiColorDisk(E, i, Type, t)
        elif Type == 4: 
            Lnu = self.PowerLaw(E, i, Type, t)    
        elif Type == 5:
            Lnu = self.QuasarTemplate(E, i, Type, t)    
        else:
            Lnu = 0.0
            
        if self.SpectrumPars.AbsorbingColumn[i] > 0:
            return Lnu * np.exp(-self.SpectrumPars.AbsorbingColumn[i] * (sigma_E(E, 0) + y * sigma_E(E, 1)))   
        else:
            return Lnu     
                
    def Spectrum(self, E, t = 0.0, only = None):
        """
        Return fraction of bolometric luminosity emitted at energy E.
        """        
        
        # Renormalize if t > 0 
        if t > 0 and t != self.last_renormalized:
            self.last_renormalized = t
            self.M = self.BlackHoleMass(t)
            self.Lbol = self.BolometricLuminosity(t)
            self.LuminosityNormalizations = self.NormalizeSpectrumComponents(t)    
        
        emission = 0
        for i, Type in enumerate(self.SpectrumPars.Type):
            if not (self.SpectrumPars.MinEnergy[i] <= E <= self.SpectrumPars.MaxEnergy[i]):
                continue
                
            if only is not None and Type != only:
                continue 
                
            emission += self.LuminosityNormalizations[i] * \
                self.Intensity(E, i, Type, t) / self.Lbol
            
        return emission
        
    def BlackBody(self, E, T = None):
        """
        Returns specific intensity of blackbody at self.T.
        """
        
        if T is None:
            T = self.T
        
        nu = E * erg_per_ev / h
        return 2.0 * h * nu**3 / c**2 / (np.exp(h * nu / k_B / T) - 1.0)
        
    def PowerLaw(self, E, i, Type, t = 0.0):    
        """
        A simple power law X-ray spectrum - this is proportional to the *energy* emitted
        at E, not the number of photons.  
        """

        return E**-self.SpectrumPars.PowerLawIndex[i]
    
    def MultiColorDisk(self, E, i, Type, t = 0.0):
        """
        Soft component of accretion disk spectra.
        """         
        
        # If t > 0, re-compute mass, inner radius, and inner temperature
        if t > 0 and self.pf.SourceTimeEvolution > 0:
            self.M = self.BlackHoleMass(t)
            self.r_in = self.DiskInnermostRadius(self.M)
            self.r_out = self.pf.SourceDiskMaxRadius * self.GravitationalRadius(self.M)
            self.T_in = self.DiskInnermostTemperature(self.M)
            self.T_out = self.DiskTemperature(self.M, self.r_out)
        
        integrand = lambda T: (T / self.T_in)**(-11. / 3.) * self.BlackBody(E, T) / self.T_in
        return quad(integrand, self.T_out, self.T_in)[0]
        
    def QuasarTemplate(self, E, i, Type, t = 0.0):
        """
        Quasar spectrum of Sazonov, Ostriker, & Sunyaev 2004.
        """    
        
        op = (E < 10)
        uv = (E >= 10) & (E < 2e3) 
        xs = (E >= 2e3) & (E < self.E_0)
        xh = (E >= self.E_0) & (E < 4e5)        
        
        if type(E) in [int, float]:
            if op:
                F = self.IR_Normalization * 1.2 * 159 * E**-0.6
            elif uv:
                F = int(uv) * self.UV_Normalization * 1.2 * E**-1.7 * np.exp(E / 2000.0)
            elif xs:
                F = self.SX_Normalization * self.A * E**-self.Alpha * np.exp(-E / self.E_1)
            elif xh:
                F = self.HX_Normalization * self.A * self.B * (1.0 + self.K * \
                    E**(self.Beta - self.Gamma)) * E**-self.Beta
            else: 
                F = 0
                
        else:
            F = np.zeros_like(E)
            F += op * self.IR_Normalization * 1.2 * 159 * E**-0.6
            F += uv * self.UV_Normalization * 1.2 * E**-1.7 * np.exp(E / 2000.0)
            F += xs * self.SX_Normalization * self.A * E**-self.Alpha * np.exp(-E / self.E_1)
            F += xh * self.HX_Normalization * self.A * self.B * (1.0 + self.K * \
                    E**(self.Beta - self.Gamma)) * E**-self.Beta
        
        return F
                            
    def NormalizeSpectrumComponents(self, t = 0):
        """
        Normalize each component of spectrum to some fraction of the bolometric luminosity.
        """
        
        Lbol = self.BolometricLuminosity(t)
        
        normalizations = np.zeros(self.N)
        for i, component in enumerate(self.SpectrumPars.Type):            
            integral, err = quad(self.Intensity, self.SpectrumPars.MinNormEnergy[i], 
                self.SpectrumPars.MaxNormEnergy[i], args = (i, component, t,))
            normalizations[i] = self.SpectrumPars.Fraction[i] * Lbol / integral
            
        return normalizations
        
    def BolometricLuminosity(self, t = 0.0, M = None):
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
            Mnow = self.BlackHoleMass(t)
            if M is not None:
                Mnow = M
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
        
    def PlotSpectrum(self, color = 'k', components = True, t = 0, normalized = True):
        import pylab as pl
        
        if not normalized:
            Lbol = self.BolometricLuminosity(t)
        else: 
            Lbol = 1
        
        E = np.logspace(np.log10(min(self.SpectrumPars.MinNormEnergy)), 
            np.log10(max(self.SpectrumPars.MaxNormEnergy)), 500)
        F = []
        
        for energy in E:
            F.append(self.Spectrum(energy, t = t))
        
        if components and self.N > 1:
            EE = []
            FF = []
            for i, component in enumerate(self.SpectrumPars.Type):
                tmpE = np.logspace(np.log10(self.SpectrumPars.MinEnergy[i]), 
                    np.log10(self.SpectrumPars.MaxEnergy[i]), 500)
                tmpF = []
                for energy in tmpE:
                    tmpF.append(self.Spectrum(energy, t = t, only = component))
                
                EE.append(tmpE)
                FF.append(tmpF)              
        
        self.ax = pl.subplot(111)    
        self.ax.loglog(E, np.array(F) * Lbol, color = color, ls = ls[0])
        
        if components and self.N > 1:
            for i in xrange(self.N):
                self.ax.loglog(EE[i], np.array(FF[i]) * Lbol, color = color, ls = ls[i + 1])
        
        self.ax.set_ylim(1e-6, 1e-2)
        self.ax.set_xlabel(r'$h\nu \ (\mathrm{eV})$')
        
        if normalized:
            self.ax.set_ylabel(r'$L_{\nu} / L_{\mathrm{bol}}$')
        else:
            self.ax.set_ylabel(r'$L_{\nu}$')
                
        pl.draw()
        
def listify(pf):
    """
    Turn any Spectrum parameter into a list, if it isn't already.
    """            
    
    Spectrum = {}
    for par in pf.keys():
        if par[0:8] != 'Spectrum':
            continue
        
        new_name = par.lstrip('Spectrum')
        if type(pf[par]) is not list:
            Spectrum[new_name] = [pf[par]]
        else:
            Spectrum[new_name] = pf[par]
            
    return Spectrum         
    
