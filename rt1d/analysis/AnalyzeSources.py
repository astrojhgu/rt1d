"""

AnalyzeSources.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sat Mar 23 19:21:46 2013

Description: 

"""

import numpy as np
import pylab as pl
from ..physics.Constants import *
from scipy.integrate import quad as integrate

allls = ['-', '--', '-.', ':']
small_number = 1e-5

class Source:
    def __init__(self, rs):
        self.rs = rs
        
    def SpectrumCDF(self, E):
        """
        Returns cumulative energy output contributed by photons at or less 
        than energy E.
        """    
        
        return integrate(self.rs.Spectrum, small_number, E)[0] 
    
    def SpectrumMedian(self, energies = None):
        """
        Compute median emission energy from spectrum CDF.
        """
        
        if energies is None:
            energies = np.linspace(self.rs.EminNorm, self.rs.EmaxNorm, 200)
        
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
        
        integrand = lambda E: self.rs.Spectrum(E) * E
        
        return integrate(integrand, self.rs.EminNorm, self.rs.EmaxNorm)[0]
                
    def PlotSpectrum(self, color='k', components=True, t=0, normalized=True,
        bins=100, ax=None, label=None, ls='-'):
        
        if not normalized:
            Lbol = self.rs.BolometricLuminosity(t)
        else: 
            Lbol = 1.
        
        E = np.logspace(np.log10(self.rs.Emin), np.log10(self.rs.Emax), bins)
        F = []
        
        for energy in E:
            F.append(self.rs.Spectrum(energy, t = t))
        
        if components and self.rs.N > 1:
            EE = []
            FF = []
            for i, component in enumerate(self.rs.SpectrumPars['type']):
                tmpE = np.logspace(np.log10(self.rs.SpectrumPars['Emin'][i]), 
                    np.log10(self.rs.SpectrumPars['Emax'][i]), bins)
                tmpF = []
                for energy in tmpE:
                    tmpF.append(self.rs.Spectrum(energy, t=t, i=i))
                
                EE.append(tmpE)
                FF.append(tmpF)
        
        if ax is None:
            ax = pl.subplot(111)
                    
        ax.loglog(E, np.array(F) * Lbol, color=color, ls=ls, 
            label=label)
        
        if components and self.rs.N > 1:
            for i in xrange(self.rs.N):
                ax.loglog(EE[i], np.array(FF[i]) * Lbol, color=color, 
                    ls=allls[i+1])
        
        ax.set_xlabel(r'$h\nu \ (\mathrm{eV})$')
        
        if normalized:
            ax.set_ylabel(r'$L_{\nu} / L_{\mathrm{bol}}$')
        else:
            ax.set_ylabel(r'$L_{\nu} \ (\mathrm{erg \ s^{-1}})$')
                
        pl.draw()
              
        return ax 
            