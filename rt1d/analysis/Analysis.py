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
from rt1d.analysis.Dataset import Dataset

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
        
        # Stuff for analytic solution
        self.Ndot = self.pf["SpectrumPhotonLuminosity"]
        self.alpha_HII = 2.6e-13 * (self.data[0].T[0] / 1.e4)**-0.85
        self.trec = 1. / self.alpha_HII / self.data[0].n_HI[-1] / s_per_myr
        self.rs = (3. * self.Ndot / 4. / np.pi / self.alpha_HII / self.data[0].n_HI[-1]**2)**(1. / 3.) / cm_per_kpc
        
        # Auto-run a few things
        self.ComputeDistributionFunctions()
        
    def StromgrenSphere(self, t):
        """
        Classical analytic solution for expansion of an HII region in an isothermal medium.  Given the time
        in seconds, will return the I-front radius in centimeters.
        """
        
        return self.rs * (1. - np.exp(-t / trec))**(1. / 3.) + self.StartRadius
        
    def LocateIonizationFront(self, dd):
        """
        Find the position of the ionization front in data dump 'dd'.
        """
        
        return np.interp(0.5, self.data[dd].x_HI, self.data[dd].r)
        
    def PlotIonizationFrontEvolution(self):
        """
        Compute analytic and numerical I-front radii vs. time and plot.
        """    
        
        pass
        
    def ComputeDistributionFunctions(self, normalize = True, Nbins = 20):
        """
        Histogram all fields.
        """            

        self.pdf = {}
        self.cdf = {}        
        self.bins = {}
        
        # Bins for neutral/ionized fractions, etc.
        self.bins['x_HI'] = self.bins['x_HII'] = self.bins['n_HI'] = self.bins['n_HII'] = np.logspace(-5, 0, Nbins)
        self.bins['T'] = np.logspace(0, 5, Nbins)
        
        self.mbins = {}
        for element in fields: self.mbins[element] = misc.binmid(self.bins[element])
        
        for element in fields:
            self.pdf[element] = []
            self.cdf[element] = []
            for dd in self.data.keys():
                exec('hist, bin_edges = np.histogram(self.data[{0}].{1}, bins = self.bins[\'{1}\'])'.format(dd, element))
                
                if normalize: norm = float(np.sum(hist))
                else: norm = 1.
                
                self.pdf[element].append(hist / norm)
                self.cdf[element].append(np.cumsum(hist) / float(np.sum(hist)))
                
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

    
    