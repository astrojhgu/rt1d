"""
Analysis.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2011-06-17.

Description: Functions to calculate various quantities from our rt1d datasets.
     
"""

import os
import numpy as np
import pylab as pl
from .Multiplot import *
from .Dataset import Dataset
from .Inspection import Inspect
from ..mods.Constants import *
from ..mods.Cosmology import *
from ..mods.Interpolate import Interpolate
from ..mods.InitializeGrid import InitializeGrid
from ..mods.RadiationSource import RadiationSource
from ..mods.SecondaryElectrons import SecondaryElectrons
from ..mods.ComputeCrossSections import PhotoIonizationCrossSection
from ..mods.InitializeIntegralTables import InitializeIntegralTables

class Analyze:
    def __init__(self, pf, retabulate = True):
        self.ds = Dataset(pf)
        self.data = self.ds.data
        self.pf = self.ds.pf        # dict
        self.g = InitializeGrid(self.pf)   
        self.cosm = Cosmology(self.pf)
        self.rs = RadiationSource(self.pf)
        self.iits = InitializeIntegralTables(self.pf)      
        self.esec = SecondaryElectrons(self.pf)         
        
        # Convenience
        self.GridDimensions = int(self.pf.GridDimensions)
        self.grid = np.arange(self.GridDimensions)
                
        # Deal with log-grid
        if self.pf.LogarithmicGrid:
            self.r = np.logspace(np.log10(self.pf.StartRadius * self.pf.LengthUnits),
                np.log10(self.pf.LengthUnits), self.GridDimensions + 1)
        else:
            self.r = np.linspace(self.pf.StartRadius * self.pf.LengthUnits, 
                self.pf.LengthUnits, self.GridDimensions + 1)
        
        self.dx = np.diff(self.r)   
        self.r = self.r[0:-1]
                
        self.Vsh = 4. * np.pi * ((self.r + self.dx)**3 - self.r**3) / 3. / cm_per_mpc**3        
                
        # Shift radii to cell-centered values
        self.r += self.dx / 2.   
                                
        # Store bins used for PDFs/CDFs
        self.bins = {}
        
        # Read integral table if it exists.
        self.tname = self.iits.DetermineTableName()
        if os.path.exists('%s/%s' % (self.ds.od, self.tname)) and retabulate:
            self.itabs = self.iits.TabulateRateIntegrals()
            self.interp = Interpolate(self.pf, self.iits)
        else:
            self.itabs = self.interp = None    
            
        # Inspect instance    
        self.inspect = Inspect(self)    
        
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
            self.Ndot = self.pf.SpectrumPhotonLuminosity
            self.alpha_HII = 2.6e-13 * (T / 1.e4)**-0.85
            self.trec = 1. / self.alpha_HII / self.data[0].n_HI[-1]                                         # s
            self.rs = (3. * self.Ndot / 4. / np.pi / self.alpha_HII / self.data[0].n_HI[-1]**2)**(1. / 3.)  # cm
        
        return self.rs * (1. - np.exp(-t / self.trec))**(1. / 3.) + self.pf.StartRadius
        
    def LocateIonizationFront(self, dd, species = 0):
        """
        Find the position of the ionization front in data dump 'dd'.
        """
        
        if species == 0:
            return np.interp(0.5, self.data[dd].x_HI, self.data[dd].r)
        else:
            return np.interp(0.5, self.data[dd].x_HeI, self.data[dd].r)
        
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

        if mp is not None: 
            self.mp = mp    
        else: 
            self.mp = multiplot(dims = (2, 1), panel_size = (1, 1), useAxesGrid = False)

        if anl: 
            self.mp.grid[0].plot(self.t / self.trec, self.ranl, linestyle = '-', color = 'k')
        
        self.mp.grid[0].plot(self.t / self.trec, self.rIF, color = color, ls = ls)
        self.mp.grid[0].set_xlim(0, max(self.t / self.trec))
        self.mp.grid[0].set_ylim(0, 1.1 * max(max(self.rIF), max(self.ranl)))
        self.mp.grid[0].set_ylabel(r'$r \ (\mathrm{kpc})$')  
        self.mp.grid[1].plot(self.t / self.trec, self.rIF / self.ranl, color = color, ls = ls)
        self.mp.grid[1].set_xlim(0, max(self.t / self.trec))
        self.mp.grid[1].set_ylim(0.95, 1.05)
        self.mp.grid[1].set_xlabel(r'$t / t_{\mathrm{rec}}$')
        self.mp.grid[1].set_ylabel(r'$r_{\mathrm{num}} / r_{\mathrm{anl}}$') 
        self.mp.grid[0].xaxis.set_ticks(np.linspace(0, 4, 5))
        self.mp.grid[1].xaxis.set_ticks(np.linspace(0, 4, 5))
        
        if mp is None: 
            self.mp.fix_ticks()  
     
    def TemperatureProfile(self, t = 10, color = 'k', ls = '-', xscale = 'linear'):
        """
        Plot radial profiles of temperature at times t (Myr).
        """  
        
        if not hasattr(self, 'ax'):
            self.ax = pl.subplot(111)
        
        self.ax.set_xscale('log')
        self.ax.set_yscale('log')
        
        for dd in self.data.keys():
            if self.data[dd].t / self.pf['TimeUnits'] != t: 
                continue
            
            exec('self.ax.loglog(self.data[%i].r / self.pf[\'LengthUnits\'], \
                self.data[%i].T, ls = \'%s\', color = \'%s\')' % (dd, dd, ls, color))                
            
        self.ax.set_xscale(xscale)    
        self.ax.set_xlabel(r'$r / L_{\mathrm{box}}$') 
        self.ax.set_ylabel(r'Temperature $(K)$')  
        pl.draw()        
        
    def IonizationProfile(self, species = 'H', t = [1, 10, 100], color = 'k', 
        annotate = False, xscale = 'linear', yscale = 'log'):
        """
        Plot radial profiles of species fraction (for H or He) at times t (Myr).
        """      
        
        if not hasattr(self, 'ax'):
            self.ax = pl.subplot(111)
        
        self.ax.set_xscale('log')
        self.ax.set_yscale('log')
        
        for dd in self.data.keys():
            if self.data[dd].t / self.pf['TimeUnits'] not in t: 
                continue
            
            if species == 'H':
                exec('self.ax.semilogy(self.data[%i].r / self.pf[\'LengthUnits\'], \
                    self.data[%i].%s, ls = \'%s\', color = \'%s\', label = r\'$x_{\mathrm{HI}}$\')' % (dd, dd, 'x_HI', '-', color))
                exec('self.ax.semilogy(self.data[%i].r / self.pf[\'LengthUnits\'], \
                    self.data[%i].%s, ls = \'%s\', color = \'%s\', label = r\'$x_{\mathrm{HII}}$\')' % (dd, dd, 'x_HII', '--', color))
            if species == 'He':
                exec('self.ax.semilogy(self.data[%i].r / self.pf[\'LengthUnits\'], \
                    self.data[%i].%s, ls = \'%s\', color = \'%s\', label = r\'$x_{\mathrm{HeI}}$\')' % (dd, dd, 'x_HeI', '-', color))
                exec('self.ax.semilogy(self.data[%i].r / self.pf[\'LengthUnits\'], \
                    self.data[%i].%s, ls = \'%s\', color = \'%s\', label = r\'$x_{\mathrm{HeII}}$\')' % (dd, dd, 'x_HeII', '--', color))
                exec('self.ax.semilogy(self.data[%i].r / self.pf[\'LengthUnits\'], \
                    self.data[%i].%s, ls = \'%s\', color = \'%s\', label = r\'$x_{\mathrm{HeIII}}$\')' % (dd, dd, 'x_HeIII', ':', color))                
            
        self.ax.set_xscale(xscale)
        self.ax.set_yscale(yscale)    
        self.ax.set_xlabel(r'$r / L_{\mathrm{box}}$') 
        self.ax.set_ylabel(r'Species Fraction')  
        
        if annotate:
            if species == 'H':
                self.ax.legend(loc = 'lower right', ncol = 2, frameon = False)
            if species == 'He':
                self.ax.legend(loc = 'lower right', ncol = 3, frameon = False)    
        
        pl.draw()
        
    def RadialProfileMovie(self, field = 'x_HI', out = 'frames', xscale = 'linear',
        title = True):
        """
        Save time-series images of 'field' to 'out' directory.
        
        field = x_HI, x_HII, x_HeI, x_HeII, x_HeIII, n_e, T
        """    
        
        if out is None:
            out = './'
        elif not os.path.exists(out):
            os.mkdir(out)
            
        if field == 'T':
            mi, ma = (1e2, 1e5)
        else:
            mi, ma = (1e-5, 1.5)
            
        ax = pl.subplot(111)    
        ax.set_xscale('log')        
        ax.set_yscale('log')  
         
        for dd in self.data.keys():
            
            exec('ax.loglog(self.data[%i].r / self.pf[\'LengthUnits\'], \
                self.data[%i].%s, ls = \'-\', color = \'k\')' % (dd, dd, field))

            ax.set_xlim(self.data[0].r[0] / self.pf.LengthUnits, 1)        
            ax.set_ylim(mi, ma)    
            ax.set_xscale(xscale) 
            
            if title:
                ax.set_title(r'$t = %g \ \mathrm{Myr}$' % self.data[dd].t / self.pf['TimeUnits'])
                    
            pl.savefig('%s/dd%s_%s.png' % (out, str(dd).zfill(4), field))                        
            ax.clear()
            
        pl.close()    
        
    def Ionization_Temperature_Movie(self, out = 'frames', xscale = 'linear', title = True):
        """
        Meant to answer Eric's question.
        """    
                
        for dd in self.data.keys():
            
            mp = multiplot(dims = (2, 1), useAxesGrid = False)
            
            exec('mp.grid[0].semilogy(self.data[%i].r / self.pf[\'LengthUnits\'], \
                self.data[%i].x_HI, ls = \'-\', color = \'k\', label = r\'$x_{\mathrm{HI}}$\')' % (dd, dd))
            exec('mp.grid[0].semilogy(self.data[%i].r / self.pf[\'LengthUnits\'], \
                self.data[%i].x_HII, ls = \'--\', color = \'k\', label = r\'$x_{\mathrm{HII}}$\')' % (dd, dd))    
            exec('mp.grid[1].semilogy(self.data[%i].r / self.pf[\'LengthUnits\'], \
                self.data[%i].T, ls = \'-\', color = \'k\')' % (dd, dd))
                
            mp.grid[0].set_xlim(self.data[0].r[0] / self.pf.LengthUnits, 1)        
            mp.grid[1].set_xlim(self.data[0].r[0] / self.pf.LengthUnits, 1)        
            mp.grid[1].set_xscale(xscale)     
            mp.grid[0].set_xscale(xscale)     
                        
            mp.grid[1].set_xlabel(r'$r / L_{\mathrm{box}}$') 
            mp.grid[0].set_ylabel(r'Species Fraction')  
            mp.grid[1].set_ylabel(r'Temperature $(K)$')
            
            mp.grid[0].set_ylim(1e-5, 1.1)
            mp.grid[1].set_ylim(1e2, 1e5)  
            
            mp.fix_ticks()
            
            mp.grid[0].legend(loc = 'lower right', ncol = 2, frameon = False)
            
            if title:
                mp.grid[0].set_title(r'$t = %g \ \mathrm{Myr}$' % (self.data[dd].t / self.pf['TimeUnits']))
            
            pl.draw()        
            pl.savefig('%s/dd%s_xT.png' % (out, str(dd).zfill(4)))                        
            pl.close()
        
    def ClumpTest(self, t = [1,3, 15], color = 'k', legend = True):
        """
        RT06 Problem #3.
        """
        
        self.mp = multiplot(dims = (2, 1), useAxesGrid = False)
        
        ct = 0
        ls = ['-', ':', '--', '-.']
        for dd in self.data.keys():
            if self.data[dd].t / self.pf.TimeUnits not in t: 
                continue
                
            this_t = int(self.data[dd].t / self.pf.TimeUnits)
        
            self.mp.grid[0].semilogy(self.data[dd].r / self.pf.LengthUnits, self.data[dd].x_HI, color = color, ls = ls[ct], 
                label = r'$t = %i \ \mathrm{Myr}$' % this_t)
            self.mp.grid[1].semilogy(self.data[dd].r / self.pf.LengthUnits, self.data[dd].T, color = color, ls = ls[ct])
            ct += 1
        
        self.mp.grid[0].set_ylim(1e-3, 1.5)
        self.mp.grid[1].set_ylim(10, 8e4)
                                                
        for i in xrange(2):
            self.mp.grid[i].set_xlim(0.6, 1.0)
                                    
        self.mp.grid[1].set_xlabel(r'$x / L_{\mathrm{box}}$')    
        self.mp.grid[0].set_ylabel('Neutral Fraction')
        self.mp.grid[1].set_ylabel(r'Temperature $(K)$')    
        self.mp.fix_ticks()
        
        if legend:
            self.mp.grid[0].legend(loc = 'lower right', frameon = False)    
                        
        pl.draw()        
        
    def CellTimeEvolution(self, cell = 0, field = 'x_HI'):
        """
        Return time evolution of cell.
        """    
        
        time = []
        value = []
        for dd in self.data.keys():
            if self.pf.CosmologicalExpansion:
                time.append(self.data[dd].z)
            else:
                time.append(self.data[dd].t)
            exec('value.append(self.data[%i].%s[%i])' % (dd, field, cell))
        
        return np.array(time), np.array(value)
        
    def CellTimeSeries(self, cell = 0, species = 0, field = 'x_HI', color = 'k', ls = '-'):
        """
        Plot cell evolution.
        """    
        
        t, val = self.CellTimeEvolution(cell = cell, field = field)
        
        if len(val.shape) > 1:
            val = zip(*val)[species]
            
        if field in ['Gamma', 'gamma']:
            t, nabs = self.CellTimeEvolution(cell = cell, field = 'nabs')
            val *= zip(*nabs)[species]
        elif field in ['Beta']:
            t, nabs = self.CellTimeEvolution(cell = cell, field = 'nabs')
            t, ne = self.CellTimeEvolution(cell = cell, field = 'n_e')
            val *= zip(*nabs)[species] * ne
        elif field in ['zeta', 'psi']:
            t, nabs = self.CellTimeEvolution(cell = cell, field = 'nabs')
            t, ne = self.CellTimeEvolution(cell = cell, field = 'n_e')
            val *= zip(*nabs)[species] * ne
        elif field in ['eta']:
            t, nion = self.CellTimeEvolution(cell = cell, field = 'nion')
            t, ne = self.CellTimeEvolution(cell = cell, field = 'n_e')
            val *= zip(*nion)[species] * ne
            
        self.ax = pl.subplot(111)
        self.ax.loglog(t / s_per_yr, val, color = color, ls = ls)  
        
        pl.draw()    
        
    def InspectRateCoefficients(self, t = 1, coeff = 'Gamma', species = 0, color = 'k', ls = '-'):
        """
        Plot given rate coefficient as function of r.
        """
        
        if species == 0:
            s = 'HI'
        elif species == 1:
            s = 'HeI'
        else:
            s = 'HeII'
            
        if coeff == 'Gamma':
            ylabel = r'$\Gamma_{\mathrm{%s}}$' % s
        elif coeff == 'gamma':
            ylabel = r'$\gamma_{\mathrm{%s}}$' % s
        elif coeff == 'Beta':
            ylabel = r'$\beta_{\mathrm{%s}}$' % s    
        elif coeff == 'k_H':
            ylabel = r'$\mathcal{H}_{\mathrm{%s}}$' % s
        elif coeff == 'zeta':
            ylabel = r'$\zeta_{\mathrm{%s}}$' % s
        elif coeff == 'eta':
            ylabel = r'$\eta_{\mathrm{%s}}$' % s 
        elif coeff == 'psi':
            ylabel = r'$\psi_{\mathrm{%s}}$' % s       
        elif coeff == 'omega':
            ylabel = r'$\omega_{\mathrm{HeIII}}$'   
        elif coeff == 'alpha':
            ylabel = r'$\alpha_{\mathrm{%s}}$' % s               
        
        self.ax = pl.subplot(111)
        for dd in self.data.keys():
            if self.data[dd].t / self.pf.TimeUnits != t: 
                continue
            
            exec('self.ax.loglog(self.data[%i].r / self.pf[\'LengthUnits\'], \
                self.data[%i].%s[%i], ls = \'%s\', color = \'%s\')' % (dd, dd, coeff, species, ls, color))                
            
        self.ax.set_xlabel(r'$r / L_{\mathrm{box}}$') 
        self.ax.set_ylabel(ylabel)         
                                
        pl.draw()      
        
    def IonizationRate(self, t = 1, species = 0, color = 'k', legend = True, plot_recomb = False):
        """
        Plot total ionization rate, and lines for primary, secondary, and collisional.
        """ 
        
        if legend and hasattr(self, 'ax'):
            legend = False
        
        for dd in self.data.keys():
            if self.data[dd].t / self.pf.TimeUnits != t: 
                continue
                            
            Gamma = self.data[dd].Gamma[:,species] * self.data[dd].nabs[species,:]
            gamma = np.sum(self.data[dd].gamma[:,species,:] * np.transpose(self.data[dd].nabs), axis = 1)
            Beta = self.data[dd].Beta[:,species] * self.data[dd].nabs[species,:] * self.data[dd].n_e            
            ion = Gamma + Beta + gamma
            
            alpha = self.data[dd].alpha[:,species] * self.data[dd].nion[species,:] * self.data[dd].n_e
            xi = self.data[dd].xi[:,species] * self.data[dd].nion[species,:] * self.data[dd].n_e
            recomb = alpha + xi    
                
        self.ax = pl.subplot(111)
        self.ax.loglog(self.data[dd].r / self.pf.LengthUnits, ion, color = color, ls = '-', label = 'Ioniz.')        
        self.ax.loglog(self.data[dd].r / self.pf.LengthUnits, Gamma, color = color, ls = '--', label = r'$\Gamma$')
        self.ax.loglog(self.data[dd].r / self.pf.LengthUnits, gamma, color = color, ls = ':', label = r'$\gamma$')
        self.ax.loglog(self.data[dd].r / self.pf.LengthUnits, Beta, color = color, ls = '-.', label = r'$\beta$')
                
        if plot_recomb:
            self.ax.loglog(self.data[dd].r / self.pf.LengthUnits, recomb, color = 'b', ls = '-', label = 'Recomb.')
            self.ax.loglog(self.data[dd].r / self.pf.LengthUnits, alpha, color = 'b', ls = '--', label = r'$\alpha$')
            self.ax.loglog(self.data[dd].r / self.pf.LengthUnits, xi, color = 'b', ls = ':', label = r'$\xi$')
        
        self.ax.set_xlabel(r'$r / L_{\mathrm{box}}$') 
        self.ax.set_ylabel(r'Ionization Rate')
        self.ax.set_ylim(0.01 * 10**np.floor(np.log10(np.min(ion))), 10**np.ceil(np.log10(np.max(ion))))
        
        if legend:
            self.ax.legend(frameon = False, ncol = 2)
        
        pl.draw()    
        
    def HeatingRate(self, t = 1, color = 'k', legend = True):
        """
        Plot total heating rate, and lines for primary, secondary, and collisional.
        """ 
        
        if legend and hasattr(self, 'ax'):
            legend = False
        
        for dd in self.data.keys():
            if self.data[dd].t / self.pf.TimeUnits != t: 
                continue
            
            x_HII = self.data[dd].x_HII
            fheat = np.zeros(self.GridDimensions)
            for i in xrange(self.GridDimensions):
                fheat[i] = self.esec.DepositionFraction(None, x_HII[i], 0)   
                
            tranabs = np.transpose(self.data[dd].nabs)             
            
            heat = fheat * np.sum(self.data[dd].k_H * tranabs, axis = 1)
            zeta = np.sum(self.data[dd].zeta * tranabs, axis = 1) * self.data[dd].n_e # collisional ionization
            eta = np.sum(self.data[dd].eta * tranabs, axis = 1) * self.data[dd].n_e  # recombination
            psi = np.sum(self.data[dd].psi * tranabs, axis = 1) * self.data[dd].n_e  # collisional excitation
            omega = np.sum(self.data[dd].omega * tranabs, axis = 1) * self.data[dd].n_e # dielectric
            cool = (zeta + eta + psi + omega)

        mi = min(np.min(heat), np.min(cool))    
        ma = max(np.max(heat), np.max(cool))    
            
        self.ax = pl.subplot(111)
        self.ax.loglog(self.data[dd].r / self.pf.LengthUnits, heat, color = 'r', ls = '-', label = r'$\mathcal{H}_{\mathrm{tot}}$')
        self.ax.loglog(self.data[dd].r / self.pf.LengthUnits, cool, color = 'b', ls = '-', label = r'$\mathcal{C}_{\mathrm{tot}}$')
        self.ax.loglog(self.data[dd].r / self.pf.LengthUnits, zeta, color = 'g', ls = '--', label = r'$\zeta$')
        self.ax.loglog(self.data[dd].r / self.pf.LengthUnits, psi, color = 'g', ls = ':', label = r'$\psi$')
        self.ax.loglog(self.data[dd].r / self.pf.LengthUnits, eta, color = 'c', ls = '--', label = r'$\eta$')
        self.ax.loglog(self.data[dd].r / self.pf.LengthUnits, omega, color = 'c', ls = ':', label = r'$\omega_{\mathrm{HeII}}$')
                
        self.ax.set_xlabel(r'$r / L_{\mathrm{box}}$') 
        self.ax.set_ylabel(r'Heating & Cooling Rate $(\mathrm{erg/s/cm^3})$')
        self.ax.set_ylim(0.001 * 10**np.floor(np.log10(mi)), 10**np.ceil(np.log10(ma)))
        
        if legend:
            self.ax.legend(frameon = False, ncol = 3, loc = 'lower right')
        
        pl.draw()    
        
        # Save heating and cooling rates
        self.heat = heat
        self.cool = cool
            
    def ComputeDistributionFunctions(self, field, normalize = True, bins = 20, volume = False):
        """
        Histogram all fields.
        """            
                
        pdf = []
        cdf = []
        icdf = []
        for dd in self.data.keys():
            exec('hist, bin_edges = np.histogram(self.data[{0}].{1}, bins = bins)'.format(dd, field))                
                            
            if volume: 
                hist = hist**3                
                            
            if normalize: 
                norm = float(np.sum(hist))
            else: 
                norm = 1.
            
            pdf.append(hist / norm)
            cdf.append(np.cumsum(hist) / float(np.sum(hist)))
            icdf.append(1. - np.array(cdf[dd]))
        
        bins = self.rebin(bin_edges)
                
        return {'bins': bins, 'pdf': pdf, 'cdf': cdf, 'icdf': icdf}
                
    def PlotDistributionFunction(self, dd, field = 'x_HI', df = 'pdf', color = 'k'):
        """
        Make nice plot of distribution functions.
        """
        
        self.ax = pl.subplot(111)
        
        if df == 'pdf': 
            hist = self.pdf
        else: 
            hist = self.cdf
        
        self.ax.plot(self.bins[field], hist[field][dd], color = color, drawstyle = 'steps-mid')
        self.ax.set_xscale('log')
        pl.draw()
        
    def rebin(self, bins, center = False):
        """
        Take in an array of bin edges (centers) and convert them to bin centers (edges).
        
            center: Input bin values refer to bin centers?
            
        """
        
        bins = np.array(bins)
        
        if center:
            result = 0
        else:
            result = np.zeros(bins.size - 1)
            for i, element in enumerate(result): 
                result[i] = (bins[i] + bins[i + 1]) / 2.
                
        return result
        
        

    
    