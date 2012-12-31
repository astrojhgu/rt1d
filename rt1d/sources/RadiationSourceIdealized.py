"""

RadiationSourceIdealized.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sun Jul 22 16:28:08 2012

Description: 

"""

import numpy as np
from ..physics.Constants import *
from scipy.integrate import quad, romberg
from ..physics.ComputeCrossSections import PhotoIonizationCrossSection as sigma_E

from ..util import parse_kwargs


np.seterr(all = 'ignore')   # exp overflow occurs when integrating BB
                            # will return 0 as it should for x large

SchaererTable = {
                "Mass": [5, 9, 15, 25, 40, 60, 80, 120, 200, 300, 400, 500, 1000], 
                "Temperature": [4.44, 4.622, 4.759, 4.85, 4.9, 4.943, 4.97, 4.981, 4.999, 5.007, 5.028, 5.029, 5.026],
                "Luminosity": [2.87, 3.709, 4.324, 4.89, 5.42, 5.715, 5.947, 6.243, 6.574, 6.819, 6.984, 7.106, 7.444]
                }

E_th = [13.6, 24.6, 54.4]
small_number = 1e-3                
big_number = 1e5
ls = ['-', '--', ':', '-.']

class RadiationSourceIdealized:
    def __init__(self, **kwargs):
        self.pf = parse_kwargs(**kwargs)
        
        # Create Source/SpectrumPars attributes    
        self.SourcePars = sort(self.pf, prefix = 'source', make_list = False)
        self.SpectrumPars = sort(self.pf, prefix = 'spectrum')
        
        # Number of spectral components
        self.N = len(self.SpectrumPars['type'])
        
        # Cast types to int to avoid indexing complaints
        self.SpectrumPars['type'] = map(int, self.SpectrumPars['type'])
        
        self._name = 'RadiationSourceIdealized'
        self.multi_freq = self.SpectrumPars['multifreq']
        self.multi_grp = self.SpectrumPars['multigrp']
                
        self.initialize()
        
    def initialize(self):    
        """
        Create attributes we need, normalize, etc.
        """
        
        self.Emin = min(self.SpectrumPars['Emin'])
        self.Emax = min(self.SpectrumPars['Emax'])
        self.EminNorm = min(self.SpectrumPars['EminNorm'])
        self.EmaxNorm = min(self.SpectrumPars['EmaxNorm'])
        
        if self.N == 1:
            self.Type = self.SpectrumPars['type'][0]
                    
        # Correct later if using multi-group approach
        self.E = np.array(self.SpectrumPars['E'])
        self.LE = np.array(self.SpectrumPars['LE'])
        self.Nfreq = len(self.E)
        
        self.last_renormalized = 0
        self.tau = self.SourcePars['lifetime'] * self.pf['TimeUnits']
        self.birth = self.SourcePars['tbirth'] * self.pf['TimeUnits']
        self.fduty = self.SourcePars['fduty']
        
        self.variable = self.fduty < 1
        if self.fduty == 1:
            self.variable = self.tau < (self.pf['StopTime'] * self.pf['TimeUnits'])
                
        self.toff = self.tau * (self.fduty**-1. - 1.)
                        
        # SourceType 0, 1, 2
        self.Lph = self.SpectrumPars['qdot']
        
        # SourceType = 1, 2
        self.T = self.SourcePars['temperature']
        
        # SourceType >= 3
        self.M = self.SourcePars['mass']
        self.M0 = self.SourcePars['mass']
        self.epsilon = self.SourcePars['eta']
        
        if 3 in self.SpectrumPars['type']:
            self.r_in = self.DiskInnermostRadius(self.M0)
            self.r_out = self.SourcePars['rmax'] * self.GravitationalRadius(self.M0)
            self.fcol = self.SpectrumPars['fcol'][self.SpectrumPars['type'].index(3)]
            self.T_in = self.DiskInnermostTemperature(self.M0)
            self.T_out = self.DiskTemperature(self.M0, self.r_out)    

        ### PUT THIS STUFF ELSEWHERE
                                 
        # Number of ionizing photons per cm^2 of surface area for BB of temperature self.T.  
        # Use to solve for stellar radius (which we need to get Lbol).  The factor of pi gets rid of the / sr units
        if self.SourcePars['type'] == 0:
            self.Lbol = np.sum(self.Lph * self.LE * self.E * erg_per_ev)
            self.Qdot = self.LE * self.Lph
            self.sigma = sigma_E(self.E)
        elif self.SourcePars['type'] in [1, 2]:
            self.LphNorm = np.pi * 2. * (k_B * self.T)**3 * \
                romberg(lambda x: x**2 / (np.exp(x) - 1.), 
                13.6 * erg_per_ev / k_B / self.T, big_number, divmax = 100) / h**3 / c**2 
            self.R = np.sqrt(self.Lph / 4. / np.pi / self.LphNorm)        
            self.Lbol = 4. * np.pi * self.R**2 * sigma_SB * self.T**4
            self.Qdot = self.Lbol * self.LE / self.E / erg_per_ev
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
        Qnorm = self.SpectrumPars['qdot'] / self.Lbol / \
            quad(lambda x: self.Spectrum(x) / x, self.EminNorm, self.EmaxNorm)[0]
        if self.pf['FrequencyAveragedCrossSections']:
            self.E = np.zeros(self.pf['FrequencyGroups'])
            self.LE = np.ones_like(self.E)
            self.Qdot = np.zeros_like(self.E)
            self.bands = self.pf['FrequencyBands']
            
            if len(self.bands) == (len(self.E) - 1):
                self.bands.append(self.Emax)
            
            for i in xrange(int(self.pf['FrequencyGroups'])):
                L = quad(lambda x: self.Spectrum(x), self.bands[i], self.bands[i + 1])[0]
                Q = quad(lambda x: self.Spectrum(x) / x, self.bands[i], self.bands[i + 1])[0]
                                
                self.E[i] = L / Q
                self.Qdot[i] = Qnorm * self.Lbol * Q
                
        # Time evolution
        if self.SourcePars['evolving']:
            self.Age = np.linspace(0, self.pf['StopTime'] * self.pf['TimeUnits'], self.pf['AgeBins'])
                                              
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
        if not self.pf['SourceISCO']:
            return 6. * self.GravitationalRadius(M)
        else:
            return self.pf['SourceISCO']     
            
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
            
    def SourceOn(self, t):
        """
        See if source is on.
        """        
        
        if not self.variable:
            return True
            
        if t <= self.tau:
            return True
            
        if self.fduty == 1:
            return False    
            
        nacc = t / (self.tau + self.toff)
        if nacc % 1 < self.fduty:
            return True
        else:
            return False
            
    def BlackHoleMass(self, t):
        """
        Compute black hole mass after t (seconds) have elapsed.  Relies on 
        initial mass self.M, and (constant) radiaitive efficiency self.epsilon.
        """        
        
        if self.variable:
            nlifetimes = int(t / (self.tau + self.toff))
            dtacc = nlifetimes * self.tau
            M0 = self.M0 * np.exp(((1.0 - self.epsilon) / self.epsilon) * dtacc / t_edd)  
            dt = t - nlifetimes * (self.tau + self.toff)
        else:
            M0 = self.M0
            dt = t
        
        return M0 * np.exp(((1.0 - self.epsilon) / self.epsilon) * dt / t_edd)         
    
    def BlackHoleAge(self, M):
        """
        Compute age of black hole based on current time, current mass, and initial mass.
        """            
        
        #if self.variable:
            
            
        return np.log(M / self.pf['SourceMass']) * (self.epsilon / (1. - self.epsilon)) * t_edd
                
    def IonizingPhotonLuminosity(self, t = 0, bin = None):
        """
        Return Qdot (photons / s) for this source at energy E.
        """
        
        if self.pf['SourceType'] in [0, 1, 2]:
            return self.Qdot[bin]
        else:
            return self.BolometricLuminosity(t) * self.LE[bin] / self.E[bin] / erg_per_ev          
              
    def Intensity(self, E, i, Type, t):
        """
        Return quantity *proportional* to fraction of bolometric luminosity emitted
        at photon energy E.  Normalization handled separately.
        """
        
        if Type == 0:
            Lnu = self.LE[0]
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
            
        if self.SpectrumPars['N'][i] > 0:
            return Lnu * np.exp(-self.SpectrumPars['N'][i] * (sigma_E(E, 0) + y * sigma_E(E, 1)))   
        else:
            return Lnu     
                
    def Spectrum(self, E, t = 0.0, only = None):
        """
        Return fraction of bolometric luminosity emitted at energy E.
        """        
        
        # Renormalize if t > 0 
        if t != self.last_renormalized:
            self.last_renormalized = t
            self.M = self.BlackHoleMass(t)
            self.r_in = self.DiskInnermostRadius(self.M)
            self.r_out = self.SpectrumPars['rmax'] * self.GravitationalRadius(self.M)
            self.T_in = self.DiskInnermostTemperature(self.M)
            self.T_out = self.DiskTemperature(self.M, self.r_out)
            self.Lbol = self.BolometricLuminosity(t)
            self.LuminosityNormalizations = self.NormalizeSpectrumComponents(t)    
        
        emission = 0
        for i, Type in enumerate(self.SpectrumPars['type']):
            if not (self.SpectrumPars['Emin'][i] <= E <= self.SpectrumPars['Emax'][i]):
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

        return E**-self.SpectrumPars['PowerLawIndex'][i]
    
    def MultiColorDisk(self, E, i, Type, t = 0.0):
        """
        Soft component of accretion disk spectra.
        """         
        
        # If t > 0, re-compute mass, inner radius, and inner temperature
        if t > 0 and self.pf['SourceTimeEvolution'] and t != self.last_renormalized:
            self.M = self.BlackHoleMass(t)
            self.r_in = self.DiskInnermostRadius(self.M)
            self.r_out = self.pf['SourceDiskMaxRadius'] * self.GravitationalRadius(self.M)
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
        for i, component in enumerate(self.SpectrumPars['type']):            
            integral, err = quad(self.Intensity, self.SpectrumPars['EminNorm'][i], 
                self.SpectrumPars['EmaxNorm'][i], args = (i, component, t,))
            normalizations[i] = self.SpectrumPars['fraction'][i] * Lbol / integral
            
        return normalizations
        
    def BolometricLuminosity(self, t = 0.0, M = None):
        """
        Returns the bolometric luminosity of a source in units of erg/s.  For accreting black holes, the 
        bolometric luminosity will increase with time, hence the optional 't' argument.
        """        
        
        if not self.variable:
            if t >= self.tau:
                return 0.0        
            
        if self.SourcePars['type'] == 0:
            return self.Lph / (np.sum(self.LE / self.E / erg_per_ev))
        
        if self.SourcePars['type'] == 1:
            return self.Lbol
        
        if self.SourcePars['type'] == 2:
            return 10**SchaererTable["Luminosity"][SchaererTable["Mass"].index(self.M)] * lsun
            
        if self.SourcePars['type'] == 3:
            if not self.SourceOn(t):
                return 0.0
                
            Mnow = self.BlackHoleMass(t)
            if M is not None:
                Mnow = M
            return self.epsilon * 4.0 * np.pi * G * Mnow * g_per_msun * m_p * c / sigma_T
    
        if self.SourcePars['type'] == 4:
            return self.pf['cX'] * 3.4e40
    
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
        
    def FrequencyAveragedBin(self, species = 0, Emin = None, Emax = None):
        """
        Bolometric luminosity / number of ionizing photons in spectrum in bandpass
        spanning interval (Emin, Emax). Returns mean photon energy and number of 
        ionizing photons in band.
        """     
        
        if Emin is None:
            Emin = max(E_th[species], self.Emin)
        if Emax is None:
            Emax = self.Emax
            
        L = self.Lbol * quad(lambda x: self.Spectrum(x), Emin, Emax)[0] 
        Q = self.Lbol * quad(lambda x: self.Spectrum(x) / x, Emin, Emax)[0] / erg_per_ev
                        
        return L / Q / erg_per_ev, Q
        
    def PlotSpectrum(self, color = 'k', components = True, t = 0, normalized = True,
        bins = 100, mp = None, label = None):
        import pylab as pl
        
        if not normalized:
            Lbol = self.BolometricLuminosity(t)
        else: 
            Lbol = 1.
        
        E = np.logspace(np.log10(min(self.SpectrumPars['MinNormEnergy'])), 
            np.log10(max(self.SpectrumPars['MaxNormEnergy'])), bins)
        F = []
        
        for energy in E:
            F.append(self.Spectrum(energy, t = t))
        
        if components and self.N > 1:
            EE = []
            FF = []
            for i, component in enumerate(self.SpectrumPars['Type']):
                tmpE = np.logspace(np.log10(self.SpectrumPars['MinEnergy'][i]), 
                    np.log10(self.SpectrumPars['MaxEnergy'][i]), bins)
                tmpF = []
                for energy in tmpE:
                    tmpF.append(self.Spectrum(energy, t = t, only = component))
                
                EE.append(tmpE)
                FF.append(tmpF)
        
        if mp is None:
            self.ax = pl.subplot(111)
        else:
            self.ax = mp
                    
        self.ax.loglog(E, np.array(F) * Lbol, color = color, ls = ls[0], label = label)
        
        if components and self.N > 1:
            for i in xrange(self.N):
                self.ax.loglog(EE[i], np.array(FF[i]) * Lbol, color = color, ls = ls[i + 1])
        
        self.ax.set_xlabel(r'$h\nu \ (\mathrm{eV})$')
        
        if normalized:
            self.ax.set_ylabel(r'$L_{\nu} / L_{\mathrm{bol}}$')
        else:
            self.ax.set_ylabel(r'$L_{\nu} \ (\mathrm{erg \ s^{-1}})$')
                
        pl.draw()
        
def sort(pf, prefix = 'spectrum', make_list = True):
    """
    Turn any Source/Spectrum parameter into a list, if it isn't already.
    """            
    
    result = {}
    for par in pf.keys():
        if par[0:len(prefix)] != prefix:
            continue
        
        new_name = par.partition('_')[-1]
        if type(pf[par]) is not list and make_list:
            result[new_name] = [pf[par]]
        else:
            result[new_name] = pf[par]
            
    return result             
        