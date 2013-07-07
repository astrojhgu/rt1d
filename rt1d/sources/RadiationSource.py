#"""
#
#RadiationSource.py
#
#Author: Jordan Mirocha
#Affiliation: University of Colorado at Boulder
#Created on: Sun Jul 22 16:28:08 2012
#
#Description: Initialize a radiation source.
#
#"""

import h5py, re
import numpy as np
from ..physics.Constants import *
from scipy.integrate import quad, romberg
from ..static.IntegralTables import IntegralTable
from ..static.InterpolationTables import LookupTable
from ..util import parse_kwargs, sort, evolve, readtab, Gauss1D, boxcar
from ..physics.CrossSections import PhotoIonizationCrossSection as sigma_E

np.seterr(all = 'ignore')   # exp overflow occurs when integrating BB
                            # will return 0 as it should for x large

E_th = [13.6, 24.6, 54.4]
small_number = 1e-3
big_number = 1e5

sptypes = {'poly':0, 'bb':1, 'mcd':2, 'pl':3, 'qso':4, 'user':5, 'toy':6,
    'line':7}
srctypes = {'test':0, 'star':1, 'bh':2, 'diffuse':3}

class RadiationSource(object):
    """ Class for creation and manipulation of radiation sources. """
    def __init__(self, grid=None, logN=None, init_tabs=True, **kwargs):
        """ 
        Initialize a radiation source object. 
    
        Parameters
        ----------
        grid: rt1d.static.Grid.Grid instance
        logN: column densities over which to tabulate integral quantities
    
        """    
        self.pf = parse_kwargs(**kwargs)
        self.grid = grid
                
        # Modify parameter file if spectrum_file provided
        self._load_spectrum()        
                                
        # Create Source/SpectrumPars attributes
        self.SourcePars = sort(self.pf, prefix='source', make_list=False)        
        self.SpectrumPars = sort(self.pf, prefix='spectrum')
                          
        # Number of spectral components
        self.N = len(self.SpectrumPars['type'])
        
        # Cast types to int to avoid indexing complaints
        for i, comp in enumerate(self.SpectrumPars['type']):
            if type(comp) is str:
                self.SpectrumPars['type'][i] = sptypes[comp]
                
        self.SpectrumPars['type'] = map(int, self.SpectrumPars['type'])
        
        # Convert source types to int
        if type(self.SourcePars['type']) is str:
            self.SourcePars['type'] = srctypes[self.SourcePars['type']]
                
        self.discrete = (self.SpectrumPars['E'][0] != None) \
                      or self.pf['optically_thin']
        self.continuous = not self.discrete
                
        # We don't allow multi-component discrete spectra...for now
        # would we ever want/need this? lines on top of continuous spectrum perhaps...
        self.multi_group = self.discrete and self.SpectrumPars['multigroup'][0] 
        self.multi_freq = self.discrete and not self.SpectrumPars['multigroup'][0] 

        # See if source emits ionizing photons (component by component)
        self.ionizing = np.array(self.SpectrumPars['Emax']) > E_LL
        # Should also be function of absorbers
        
        # Just a set of power-laws? (only for photons below 13.6 eV)
        self.plseries = np.all(np.array(self.SpectrumPars['type']) == sptypes['pl']) \
                    and np.all(np.array(self.SpectrumPars['Emax']) <= E_LL)

        self._initialize()
        if init_tabs:   # Set default to False?
            self.create_integral_table(logN=logN) 
            
    def _initialize(self):
        """
        Create attributes we need, normalize, etc.
        """
        
        if self.SourcePars['type'] == 3:
            self.ionization_rate = evolve(self.SourcePars['ion'])
            self.secondary_ionization_rate = evolve(self.SourcePars['ion2'])
            self.heating_rate = evolve(self.SourcePars['heat'])
        
        self.Emin = min(self.SpectrumPars['Emin'])
        self.Emax = max(self.SpectrumPars['Emax'])
        self.logEmin = np.log10(self.Emin)
        self.logEmax = np.log10(self.Emax)
                
        for i, comp in enumerate(self.SpectrumPars['type']):
            if self.SpectrumPars['EminNorm'][i] == None:
                self.SpectrumPars['EminNorm'][i] = self.SpectrumPars['Emin'][i]
            if self.SpectrumPars['EmaxNorm'][i] == None:
                self.SpectrumPars['EmaxNorm'][i] = self.SpectrumPars['Emax'][i]    
        
        self.EminNorm = self.SpectrumPars['EminNorm']
        self.EmaxNorm = self.SpectrumPars['EmaxNorm']
                         
        # Correct later if using multi-group approach
        self.E = np.array(self.SpectrumPars['E'])
        self.LE = np.array(self.SpectrumPars['LE'])
        self.Nfreq = len(self.E)
                
        self.last_renormalized = 0
        self.tau = self.SourcePars['lifetime'] * self.pf['time_units']
        self.birth = self.SourcePars['tbirth'] * self.pf['time_units']
        self.fduty = self.SourcePars['fduty']
        
        self.variable = self.fduty < 1
        if self.fduty == 1:
            self.variable = self.tau < (self.pf['stop_time'] * self.pf['time_units'])
                
        self.toff = self.tau * (self.fduty**-1. - 1.)
                        
        # For stars, normalize SED to ionizing photon luminosity
        self.Q = self.SourcePars['qdot']
        self.T = self.SourcePars['temperature']
        
        # For BHs, we'll need to know the mass and radiative efficiency
        self.M = self.SourcePars['mass']
        self.M0 = self.SourcePars['mass']
        self.epsilon = self.SourcePars['eta']        

        # Source specific initialization
        if self.SourcePars['type'] == 1:
            self._init_star()
        elif self.SourcePars['type'] == 2 and 2 in self.SpectrumPars['type']:
            self._init_bh()    
        else:
            self.Lbol = self.BolometricLuminosity(0.0)    
             
        # Normalize spectrum
        self.LuminosityNormalizations = self.NormalizeSpectrumComponents(0.0)
        
        if self.pf['optically_thin']:
            self.E = self.hnu_bar    
            
    def _init_multi_freq(self):
        pass
        
    def _init_star(self):
        # Number of ionizing photons per cm^2 of surface area for BB of 
        # temperature self.T. Use to solve for stellar radius (which we need 
        # to get Lbol).  The factor of pi gets rid of the / sr units
        self.QNorm = np.pi * 2. * (k_B * self.T)**3 * \
                romberg(lambda x: x**2 / (np.exp(x) - 1.), 
                13.6 * erg_per_ev / k_B / self.T, big_number, divmax = 100) / h**3 / c**2             
        self.R = np.sqrt(self.Q / 4. / np.pi / self.QNorm)        
        self.Lbol = 4. * np.pi * self.R**2 * sigma_SB * self.T**4
        
    def _init_bh(self):
        self.r_in = self.DiskInnermostRadius(self.M0)
        self.r_out = self.SourcePars['rmax'] * self.GravitationalRadius(self.M0)
        self.fcol = self.SpectrumPars['fcol'][self.SpectrumPars['type'].index(2)]
        self.T_in = self.DiskInnermostTemperature(self.M0)
        self.T_out = self.DiskTemperature(self.M0, self.r_out)
        self.Lbol = self.BolometricLuminosity(0.0)
        
    def _init_agn_template(self):
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
                                                               
    def _load_spectrum(self):
        """ Modify a few parameters if spectrum_file provided. """
        
        fn = self.pf['spectrum_file']
        
        if fn is None:
            return
            
        # Read spectrum - expect hdf5 with (at least) E, LE, and t datasets.    
        if re.search('.hdf5', fn):    
            f = h5py.File(fn)
            try:
                self.pf['spectrum_t'] = f['t'].value
            except:
                self.pf['spectrum_t'] = None
                self.pf['spectrum_evolving'] = False
                    
            self.pf['spectrum_E'] = f['E'].value
            self.pf['spectrum_LE'] = f['LE'].value
            f.close()
            
            if len(self.pf['spectrum_LE'].shape) > 1 \
                and not self.pf['spectrum_evolving']:
                self.pf['spectrum_LE'] = self.pf['spectrum_LE'][0]
        else: 
            spec = readtab(fn)
            if len(spec) == 2:
                self.pf['spectrum_E'], self.pf['spectrum_LE'] = spec
            else:
                self.pf['spectrum_E'], self.pf['spectrum_LE'], \
                    self.pf['spectrum_t'] = spec
                    
    def create_integral_table(self, logN=None):
        """
        Take tables and create interpolation functions.
        """
        
        if self.discrete or self.SourcePars['type'] == 3:
            return
        
        if self.SourcePars['table'] is None:
            # Overide defaults if supplied - this is dangerous
            if logN is not None:
                self.pf.update({'spectrum_dlogN': [np.diff(tmp) for tmp in logN]})
                self.pf.update({'spectrum_logNmin': [np.min(tmp) for tmp in logN]})
                self.pf.update({'spectrum_logNmax': [np.max(tmp) for tmp in logN]})
            
            # Tabulate away!            
            self.tab = IntegralTable(self.pf, self, self.grid, logN)
            self.tabs = self.tab.TabulateRateIntegrals()
        else:
            self.tab = IntegralTable(self.pf, self, self.grid, logN)
            self.tabs = self.tab.load(self.SourcePars['table'])
        
        self.setup_interp()
        
    def setup_interp(self):            
        self.tables = {}
        for tab in self.tabs:
            self.tables[tab] = \
                LookupTable(self.pf, tab, self.tab.logN, self.tabs[tab], 
                    self.tab.logx, self.tab.t)                 
    
    @property
    def sigma(self):
        """
        Compute bound-free absorption cross-section for all frequencies.
        """    
        if not self.discrete:
            return None
        if not hasattr(self, '_sigma_all'):
            self._sigma_all = np.array(map(sigma_E, self.E))
        
        return self._sigma_all
        
    @property
    def Qdot(self):
        """
        Returns number of photons emitted (s^-1) at all frequencies.
        """    
        if not hasattr(self, '_Qdot_all'):
            self._Qdot_all = self.Lbol * self.LE / self.E / erg_per_ev
        
        return self._Qdot_all
        
    @property
    def hnu_bar(self):
        """
        Average ionizing (per absorber) photon energy in eV.
        """
        if not hasattr(self, '_hnu_bar_all'):
            self._hnu_bar_all = np.zeros_like(self.grid.zeros_absorbers)
            self._qdot_bar_all = np.zeros_like(self.grid.zeros_absorbers)
            for i, absorber in enumerate(self.grid.absorbers):
                self._hnu_bar_all[i], self._qdot_bar_all[i] = \
                    self.FrequencyAveragedBin(absorber=absorber)
            
        return self._hnu_bar_all
    
    @property
    def qdot_bar(self):
        """
        Average ionizing photon luminosity (per absorber) in s^-1.
        """
        if not hasattr(self, '_qdot_bar_all'):
            hnu_bar = self.hnu_bar
            
        return self._qdot_bar_all   
    
    @property
    def sigma_bar(self):
        """
        Frequency averaged cross section (single bandpass).
        """
        if not hasattr(self, '_sigma_bar_all'):
            self._sigma_bar_all = np.zeros_like(self.grid.zeros_absorbers)
            for i, absorber in enumerate(self.grid.absorbers):
                integrand = lambda x: self.Spectrum(x) \
                    * self.grid.bf_cross_sections[absorber](x) / x
                    
                self._sigma_bar_all[i] = self.Lbol \
                    * quad(integrand, self.grid.ioniz_thresholds[absorber], 
                      self.Emax)[0] / self.qdot_bar[i] / erg_per_ev
            
        return self._sigma_bar_all
    
    @property
    def sigma_tilde(self):
        if not hasattr(self, '_sigma_tilde_all'):
            self._sigma_tilde_all = np.zeros_like(self.grid.zeros_absorbers)
            for i, absorber in enumerate(self.grid.absorbers):
                integrand = lambda x: self.Spectrum(x) \
                    * self.grid.bf_cross_sections[absorber](x)
                self._sigma_tilde_all[i] = quad(integrand, 
                    self.grid.ioniz_thresholds[absorber], self.Emax)[0] \
                    / self.fLbol_ionizing[i]
        
        return self._sigma_tilde_all
        
    @property
    def fLbol_ionizing(self):
        """
        Fraction of bolometric luminosity emitted above all ionization
        thresholds.
        """
        if not hasattr(self, '_fLbol_ioniz_all'):
            self._fLbol_ioniz_all = np.zeros_like(self.grid.zeros_absorbers)
            for i, absorber in enumerate(self.grid.absorbers):
                self._fLbol_ioniz_all[i] = quad(self.Spectrum, 
                    self.grid.ioniz_thresholds[absorber], self.Emax)[0]
                    
        return self._fLbol_ioniz_all
        
    @property
    def Gamma_bar(self):
        """
        Return ionization rate (as a function of radius) assuming optical 
        depth to cells and of cells is small.
        """
        if not hasattr(self, '_Gamma_bar_all'):
            self._Gamma_bar_all = \
                np.zeros([self.grid.dims, self.grid.N_absorbers])
            for i, absorber in enumerate(self.grid.absorbers):
                self._Gamma_bar_all[..., i] = self.Lbol * self.sigma_bar[i] \
                    * self.fLbol_ionizing[i] / 4. / np.pi / self.grid.r_mid**2 \
                    / self.hnu_bar[i] / erg_per_ev
                    
        return self._Gamma_bar_all
    
    @property
    def gamma_bar(self):
        """
        Return ionization rate (as a function of radius) assuming optical 
        depth to cells and of cells is small.
        """
        if not hasattr(self, '_gamma_bar_all'):
            self._gamma_bar_all = \
                np.zeros([self.grid.dims, self.grid.N_absorbers, 
                    self.grid.N_absorbers])
                    
            if not self.pf['secondary_ionization']:
                return self._gamma_bar_all
                    
            for i, absorber in enumerate(self.grid.absorbers):
                for j, otherabsorber in enumerate(self.grid.absorbers):
                    self._gamma_bar_all[..., i, j] = self.Gamma_bar[j] \
                        * (self.hnu_bar[j] * self.sigma_tilde[j] \
                        /  self.hnu_bar[i] / self.sigma_bar[j] \
                        - self.grid.ioniz_thresholds[otherabsorber] \
                        / self.grid.ioniz_thresholds[absorber])
                    
        return self._gamma_bar_all
    
    @property
    def Heat_bar(self):
        """
        Return ionization rate (as a function of radius) assuming optical 
        depth to cells and of cells is small.
        """
        if not hasattr(self, '_Heat_bar_all'):
            self._Heat_bar_all = \
                np.zeros([self.grid.dims, self.grid.N_absorbers])
            for i, absorber in enumerate(self.grid.absorbers):
                self._Heat_bar_all[..., i] = self.Gamma_bar[..., i] \
                    * erg_per_ev * (self.hnu_bar[i] * self.sigma_tilde[i] \
                    / self.sigma_bar[i] - self.grid.ioniz_thresholds[absorber])
                    
        return self._Heat_bar_all
                    
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
        return self.pf['source_isco'] * self.GravitationalRadius(M)
            
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
            
        if t < self.tau:
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
                    
        return np.log(M / self.pf['source_mass']) * (self.epsilon / (1. - self.epsilon)) * t_edd
                
    def IonizingPhotonLuminosity(self, t = 0, bin = None):
        """
        Return Qdot (photons / s) for this source at energy E.
        """
        
        if self.pf['source_type'] in [0, 1, 2]:
            return self.Qdot[bin]
        else:
            # Currently only BHs have a time-varying bolometric luminosity
            return self.BolometricLuminosity(t) * self.LE[bin] / self.E[bin] / erg_per_ev          
              
    def Intensity(self, E, i, Type, t=0):
        """
        Return quantity *proportional* to fraction of bolometric luminosity emitted
        at photon energy E.  Normalization handled separately.
        """
        
        if Type == 1:
            Lnu = self.BlackBody(E)
        elif Type == 2:
            Lnu = self.MultiColorDisk(E, i, Type, t)
        elif Type == 3: 
            Lnu = self.PowerLaw(E, i, t)    
        elif Type == 4:
            Lnu = self.QuasarTemplate(E, i, Type, t)
        elif Type == 7:
            Lnu = self.SpectralLine(E, i, Type, t)
        else:
            Lnu = 0.0
            
        if self.SpectrumPars['logN'][i] > 0:
            return Lnu * np.exp(-10.**self.SpectrumPars['logN'][i] \
                * (sigma_E(E, 0) + y * sigma_E(E, 1)))   
        else:
            return Lnu     
                
    def Spectrum(self, E, t=0.0, only=None):
        r"""
        Return fraction of bolometric luminosity emitted at energy E.
        
        Elsewhere denoted as :math:`I_{\nu}`, normalized such that
        :math:`\int I_{\nu} d\nu = 1`
        
        Parameters
        ----------
        E: float
            Emission energy in eV
        t: float
            Time in seconds since source turned on.    
                    
        Returns
        -------
        Fraction of bolometric luminosity emitted at E in units of eV\ :sup:`-1`\ 
                
        """       
               
        emission = 0
        for i, Type in enumerate(self.SpectrumPars['type']):
            if not self.SpectrumPars['extrapolate'][i]:
                if not (self.SpectrumPars['Emin'][i] <= E <= self.SpectrumPars['Emax'][i]):
                    continue
                
            if only is not None and Type != only:
                continue 
                
            emission += self.LuminosityNormalizations[i] * \
                self.Intensity(E, i, Type, t) / self.Lbol
            
        return emission
        
    def BlackBody(self, E, T=None):
        """
        Returns specific intensity of blackbody at self.T.
        """
        
        if T is None:
            T = self.T
                    
        nu = E * erg_per_ev / h
        return 2.0 * h * nu**3 / c**2 / (np.exp(h * nu / k_B / T) - 1.0)
        
    def PowerLaw(self, E, i, t=0.0):    
        """
        A simple power law X-ray spectrum - this is proportional to the *energy* emitted
        at E, not the number of photons.  
        """

        return E**self.SpectrumPars['alpha'][i]
    
    def MultiColorDisk(self, E, i, Type, t=0.0):
        """
        Soft component of accretion disk spectra.
        """         
        
        # If t > 0, re-compute mass, inner radius, and inner temperature
        if t > 0 and self.SpectrumPars['evolving'] and t != self.last_renormalized:
            self.M = self.BlackHoleMass(t)
            self.r_in = self.DiskInnermostRadius(self.M)
            self.r_out = self.pf['source_rmax'] * self.GravitationalRadius(self.M)
            self.T_in = self.DiskInnermostTemperature(self.M)
            self.T_out = self.DiskTemperature(self.M, self.r_out)
                    
        integrand = lambda T: (T / self.T_in)**(-11. / 3.) \
            * self.BlackBody(E, T) / self.T_in
        return quad(integrand, self.T_out, self.T_in)[0]
        
    def QuasarTemplate(self, E, i, Type, t=0):
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
        
    def SpectralLine(self, E, i, Type, t=0):
        line_pars = np.array([0., 1., self.SpectrumPars['linecenter'][i], 
            self.SpectrumPars['linewidth'][i]])    
        return Gauss1D(E, line_pars)   
                            
    def NormalizeSpectrumComponents(self, t=0):
        """
        Normalize each component of spectrum to some fraction of the 
        bolometric luminosity.
        """
        
        # If diffuse source of constant ionizing/heating background,
        # source needs no detailed spectrum.
        if self.SpectrumPars['type'] == [6]:
            return np.ones(self.N)
                    
        Lbol = self.BolometricLuminosity(t)

        normalizations = np.zeros(self.N)
        
        # Series of power-laws - enforce continuity
        if self.plseries:
            
            # Stitch Lyman-series PLs together
            for i, component in enumerate(self.SpectrumPars['type']):
                if i == 0:
                    normalizations[i] = 1.0
                else:
                    alpha_diff = self.SpectrumPars['alpha'][i-1] \
                               - self.SpectrumPars['alpha'][i]
                    normalizations[i] = \
                        self.SpectrumPars['Emin'][i]**alpha_diff
                                
            # Normalize them - can do these integrals analytically
            integral = 0.0
            for i, component in enumerate(self.SpectrumPars['type']):
                if self.SpectrumPars['alpha'][i] == 0.:
                    tmp = (self.EmaxNorm[i] \
                        - self.EminNorm[i])
                elif self.SpectrumPars['alpha'][i] == -1.:
                    tmp = np.log(self.EmaxNorm[i] \
                        / self.EminNorm[i])
                else:
                    al = self.SpectrumPars['alpha'][i]
                    tmp = (self.EmaxNorm[i]**(al+1) \
                        - self.EminNorm[i]**(al+1)) / (al+1)
                
                integral += tmp * normalizations[i]
                
            normalizations *= (Lbol / integral)
                
        # General spectrum, continuity of components not required
        else:
            for i, component in enumerate(self.SpectrumPars['type']):
                if self.SpectrumPars['normed_by'][i] == 'energy':                    
                    integral, err = quad(self.Intensity,
                        self.EminNorm[i], self.EmaxNorm[i], 
                        args=(i, component, t,))
                    
                    normalizations[i] = self.SpectrumPars['fraction'][i] * Lbol \
                        / integral
                else:
                    integral, err = quad(lambda EE: self.Intensity(EE) / EE,
                        self.EminNorm[i], self.EmaxNorm[i], 
                        args=(i, component, t,))
                
                    normalizations[i] = self.SpectrumPars['qdot'][i] * Lbol \
                        / integral
                            
        return normalizations
        
    def BolometricLuminosity(self, t=0.0, M=None):
        """
        Returns the bolometric luminosity of a source in units of erg/s.  
        For accreting black holes, the bolometric luminosity will increase 
        with time, hence the optional 't' and 'M' arguments.
        """        
        
        if not self.variable:
            if t >= self.tau:
                return 0.0        
            
        if self.SourcePars['type'] == 0:
            return self.Q / (np.sum(self.LE / self.E / erg_per_ev))
        
        if self.SourcePars['type'] == 1:
            return self.Lbol
        
        if self.SourcePars['type'] == 2:
            if not self.SourceOn(t):
                return 0.0
                
            Mnow = self.BlackHoleMass(t)
            if M is not None:
                Mnow = M
            return self.epsilon * 4.0 * np.pi * G * Mnow * g_per_msun * m_p \
                * c / sigma_T
        
        if self.SourcePars['type'] == 3:
            return 1.0        
                
    def FrequencyAveragedBin(self, absorber='h_1', Emin=None, Emax=None,
        energy_weighted=False):
        """
        Bolometric luminosity / number of ionizing photons in spectrum in bandpass
        spanning interval (Emin, Emax). Returns mean photon energy and number of 
        ionizing photons in band.
        """     
        
        if Emin is None:
            Emin = max(self.grid.ioniz_thresholds[absorber], 
                np.array(self.SpectrumPars['Emin'])[self.ionizing])
        if Emax is None:
            Emax = self.Emax
            
        if energy_weighted:
            f = lambda x: x
        else:
            f = lambda x: 1.0    
            
        L = self.Lbol * quad(lambda x: self.Spectrum(x) * f(x), Emin, Emax)[0] 
        Q = self.Lbol * quad(lambda x: self.Spectrum(x) * f(x) / x, Emin, 
            Emax)[0] / erg_per_ev
                        
        return L / Q / erg_per_ev, Q            

    def dump(self, fn, bins=100, format='hdf5'):
        """ Dump spectrum to given output format. """

        pass

