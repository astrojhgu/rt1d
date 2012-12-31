"""

RadiationSourceFromFile.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sun Jul 22 16:27:37 2012

Description: RadiationSource class used to handle user-defined SEDs supplied
via ASCII or HDF5 file.

"""

import h5py
import numpy as np

from .InitializeIntegralTables import *

class RadiationSourceFromFile:
    def __init__(self, pf):
        self.pf = pf
        
        # Create SpectrumPars attribute    
        self.SpectrumPars = listify(pf)
        self.N = len(self.SpectrumPars['Type'])
        
        # Cast types to int to avoid indexing complaints
        self.SpectrumPars['Type'] = map(int, self.SpectrumPars['Type'])   
        
        self._name = 'RadiationSourceFromFile'
        
    def initialize(self):
        """
        Create attributes we need, normalize, etc.
        """
            
        self.fn = self.pf['SpectrumFile']    
        self.tau = self.pf['SourceLifetime'] * self.pf['TimeUnits']
                
        # Read spectrum - expect hdf5 with (at least) E, L_E, and time_yr datasets.
        f = h5py.File(self.fn)
        self.E = f['E'].value
        self.t = self.Age = f['time_yr'].value * s_per_yr
        self.maxAge = np.max(self.Age)
        self.Nt = len(self.t)
        self.L_E = f['L_E'].value
        
        self.Emin = np.min(self.E)
        self.Emax = np.max(self.E)
        
        # Threshold indices
        self.i_Eth = np.zeros(3)
        for i, energy in enumerate(E_th):
            loc = np.argmin(np.abs(energy - self.E))
            
            if self.E[loc] < energy:
                loc += 1
            
            self.i_Eth[i] = loc
        
        self.Lbol = self.BolometricLuminosity(0)
        
    def SourceOn(self, t):
        if t <= self.maxAge:
            return True
        else:
            print 'WARNING: Current time lies outside bounds of spectrum table. Source now OFF.'
            return False
        
    def Spectrum(self, E = None, t = 0.0):
        """
        Return specific luminosity.
        """
        
        i = self.get_time_index(t)        
        return self.L_E[i] / self.BolometricLuminosity(t = t)
        
    def Intensity(self, E = None):
        i = self.get_time_index(t)
        return self.L_E[i]
        
    def BolometricLuminosity(self, t = 0.0):
        i = self.get_time_index(t)
        return np.trapz(self.L_E[i], self.E)
        
    def get_time_index(self, t):
        if self.pf['SourceTimeEvolution']:
            i = np.argmin(np.abs(t - self.t))
            return max(min(i, self.Nt - 2), 0)
        else:
            return 0
            
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
            
        i1 = np.argmin(np.abs(Emin - self.E))
        i2 = np.argmin(np.abs(Emax - self.E))            
        
        L = np.trapz(self.L_E[i1:i2], self.E[i1:i2])    
        Q = np.trapz(self.L_E[i1:i2] / self.E[i1:i2], self.E[i1:i2]) / erg_per_ev  
            
        return L / Q / erg_per_ev, Q
        
    def PlotSpectrum(self, color = 'k', components = True, t = 0, normalized = True,
        bins = 100, mp = None, ls = '-', label = None):
        import pylab as pl        
                
        if normalized:
            Lbol = self.BolometricLuminosity(t)
        else: 
            Lbol = 1.
        
        i = self.get_time_index(t)
        E = self.E
        F = self.L_E[i] / Lbol
        
        if mp is None:
            self.ax = pl.subplot(111)
        else:
            self.ax = mp
                    
        self.ax.loglog(E, np.array(F), color = color, ls = ls, label = label)
        
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
            