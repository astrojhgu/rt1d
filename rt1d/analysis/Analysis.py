"""
Analysis.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2011-06-17.

Description: Functions to calculate various quantities from our rt1d datasets.
     
"""

import misc
import numpy as np
import pylab as pl
from constants import *
from multiplot import *
from rt1d.analysis.Dataset import Dataset
from rt1d.mods.ComputeCrossSections import PhotoIonizationCrossSection

fields = ['T', 'n_HI', 'n_HII', 'x_HI', 'x_HII']

class Analyze:
    def __init__(self, pf):
        self.ds = Dataset(pf)
        self.data = self.ds.data
        self.pf = self.ds.pf
        
        # Convenience
        self.grid = np.arange(self.pf['GridDimensions'])
        self.r = self.pf['LengthUnits'] * self.grid / self.pf['GridDimensions']
        self.StartRadius = self.pf["StartRadius"] * self.pf['LengthUnits'] / cm_per_kpc
        
        # Auto-run a few things
        self.SetupBins()
        self.ComputeDistributionFunctions()
        
    def SetupBins(self, N = 20, scale = 'log', xmin = -5, xmax = 0, Tmin = 0, Tmax = 5):
        """
        Create bins to use for our PDF/CDF analysis.
        """    
        
        self.bins = {}
        
        # Bins for neutral/ionized fractions, etc.
        if scale == 'log':
            self.bins['x_HI'] = self.bins['x_HII'] = self.bins['n_HI'] = self.bins['n_HII'] = np.logspace(xmin, xmax, N)
            self.bins['T'] = np.logspace(Tmin, Tmax, N)
        else:
            self.bins['x_HI'] = self.bins['x_HII'] = self.bins['n_HI'] = self.bins['n_HII'] = np.linspace(xmin, xmax, N)
            self.bins['T'] = np.linspace(Tmin, Tmax, N)
        
        self.mbins = {}
        for element in fields: self.mbins[element] = misc.binmid(self.bins[element])
        
    def StromgrenSphere(self, t, sol = 0, T0 = None):
        """
        Classical analytic solution for expansion of an HII region in an isothermal medium.  Given the time
        in seconds, will return the I-front radius in centimeters.
        
        Future: sol = 1 will be the better "analytic" solution.
        """
        
        # Stuff for analytic solution
        if sol == 0:
            if T0 is not None: T = T0
            else: T = self.data[0].T[0]
            self.Ndot = self.pf["SpectrumPhotonLuminosity"]
            self.alpha_HII = 2.6e-13 * (T / 1.e4)**-0.85
            self.trec = 1. / self.alpha_HII / self.data[0].n_HI[-1]                                         # s
            self.rs = (3. * self.Ndot / 4. / np.pi / self.alpha_HII / self.data[0].n_HI[-1]**2)**(1. / 3.)  # cm
        
        return self.rs * (1. - np.exp(-t / self.trec))**(1. / 3.) + self.StartRadius
        
    def LocateIonizationFront(self, dd):
        """
        Find the position of the ionization front in data dump 'dd'.
        """
        
        return np.interp(0.5, self.data[dd].x_HI, self.data[dd].r)
        
    def ComputeIonizationFrontEvolution(self, T0 = None):
        """
        Find the position of the I-front at all times, and compute value of analytic solution.
        """    
        
        # First locate I-front for all data dumps and compute analytic solution
        self.t = np.zeros(len(self.data) - 1) # Exclude dd0000
        self.rIF = np.zeros_like(self.t)
        self.ranl = np.zeros_like(self.t)
        for i, dd in enumerate(self.data.keys()[1:]): 
            self.t[i] = self.data[dd].t
            self.rIF[i] = self.LocateIonizationFront(dd) / cm_per_kpc
            self.ranl[i] = self.StromgrenSphere(self.data[dd].t, T0 = T0) / cm_per_kpc
                
    def PlotIonizationFrontEvolution(self, mp = None, anl = True, T0 = None, color = 'k', ls = '--'):
        """
        Compute analytic and numerical I-front radii vs. time and plot.
        """    

        self.ComputeIonizationFrontEvolution(T0 = T0)

        if mp is not None: self.mp = mp    
        else: self.mp = multiplot(dims = (2, 1), panel_size = (0.5, 1))

        if anl: self.mp.axes[0].plot(self.t / self.trec, self.ranl, linestyle = '-', color = 'k')
        
        self.mp.axes[0].plot(self.t / self.trec, self.rIF, color = color, ls = ls)
        self.mp.axes[0].set_xlim(0, max(self.t / self.trec))
        self.mp.axes[0].set_ylim(0, 1.1 * max(max(self.rIF), max(self.ranl)))
        self.mp.axes[0].set_ylabel(r'$r \ (\mathrm{kpc})$')  
        self.mp.axes[1].plot(self.t / self.trec, self.rIF / self.ranl, color = color, ls = ls)
        self.mp.axes[1].set_xlim(0, max(self.t / self.trec))
        self.mp.axes[1].set_ylim(0.95, 1.05)
        self.mp.axes[1].set_xlabel(r'$t / t_{\mathrm{rec}}$')
        self.mp.axes[1].set_ylabel(r'$r/r_{\mathrm{anl}}$') 
        self.mp.axes[0].xaxis.set_ticks(np.linspace(0, 4, 5))
        self.mp.axes[1].xaxis.set_ticks(np.linspace(0, 4, 5))
        
        if mp is None: self.mp.fix_ticks()    
                     
    def ComputeDistributionFunctions(self, normalize = True, Nbins = 20):
        """
        Histogram all fields.
        """            

        self.pdf = {}
        self.cdf = {}
        self.icdf = {}  # 1 - CDF
        
        for element in fields:
            self.pdf[element] = []
            self.cdf[element] = []
            self.icdf[element] = []
            for dd in self.data.keys():
                exec('hist, bin_edges = np.histogram(self.data[{0}].{1}, bins = self.bins[\'{1}\'])'.format(dd, element))
                
                if normalize: norm = float(np.sum(hist))
                else: norm = 1.
                
                self.pdf[element].append(hist / norm)
                self.cdf[element].append(np.cumsum(hist) / float(np.sum(hist)))
            
            self.icdf[element].append(1. - np.array(self.cdf[element]))
                
    def PlotDistributionFunction(self, dd, field = 'x_HI', df = 'pdf', color = 'k'):
        """
        Make nice plot of distribution functions.
        """
        
        self.ax = pl.subplot(111)
        
        if df == 'pdf': hist = self.pdf
        else: hist = self.cdf
        
        self.ax.plot(self.mbins[field], hist[field][dd], color = color, drawstyle = 'steps-mid')
        self.ax.set_xscale('log')
        pl.draw()
        
    
        
        

    
    