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
from ..physics.Constants import *


#from .Dataset import Dataset
#from .Inspection import Inspect
#from ..mods.Constants import *
#from ..mods.Cosmology import *
#from ..mods.Interpolate import Interpolate
#from ..mods.InitializeGrid import InitializeGrid
#from ..mods.RadiationSources import RadiationSources
#from ..mods.SecondaryElectrons import SecondaryElectrons
#from ..mods.ComputeCrossSections import PhotoIonizationCrossSection
#from ..mods.InitializeIntegralTables import InitializeIntegralTables

linestyles = ['-', '--', ':', '-.']

class DummyCheckpoints:
    def __init__(self):
        pass

class Analyze:
    def __init__(self, checkpoints):
        
        if type(checkpoints) is str:
            import h5py
            f = h5py.File(checkpoints)
            
            pass
        
        self.checkpoints = checkpoints
        self.pf = checkpoints.pf
        self.grid = checkpoints.grid
        self.data = checkpoints.data
        
    def StromgrenSphere(self, t, sol = 0, T0 = None):
        """
        Classical analytic solution for expansion of an HII region in an 
        isothermal medium.  Given the time in seconds, will return the I-front 
        radius in centimeters.
        
        Future: sol = 1 will be the better "analytic" solution.
        """
        
        # Stuff for analytic solution
        if sol == 0:
            if T0 is not None: 
                T = T0
            else: 
                T = self.data[0]['T'][0]
                
            n_H = self.grid.n_H[0]
            self.Qdot = self.pf['spectrum_qdot']
            self.alpha_HII = 2.6e-13 * (T / 1.e4)**-0.85
            self.trec = 1. / self.alpha_HII / self.data[0]['h_1'][0] / n_H # s
            self.rs = (3. * self.Qdot \
                    / 4. / np.pi / self.alpha_HII / n_H**2)**(1. / 3.)  # cm
        
        return self.rs * (1. - np.exp(-t / self.trec))**(1. / 3.) + self.pf['start_radius']
        
    def LocateIonizationFront(self, dd, species = 0):
        """
        Find the position of the ionization front in data dump 'dd'.
        """
        
        if species == 0:
            return np.interp(0.5, self.data[dd]['h_1'], self.grid.r_mid)
        else:
            return np.interp(0.5, self.data[dd]['he_1'], self.grid.r_mid)
        
    def ComputeIonizationFrontEvolution(self, T0 = None):
        """
        Find the position of the I-front at all times, and compute value of analytic solution.
        """    
                
        # First locate I-front for all data dumps and compute analytic solution
        self.t = np.zeros(len(self.data) - 1) # Exclude dd0000
        self.rIF = np.zeros_like(self.t)
        self.ranl = np.zeros_like(self.t)
        for i, dd in enumerate(self.data.keys()[1:]): 
            self.t[i] = self.data[dd]['time']
            self.rIF[i] = self.LocateIonizationFront(dd) / cm_per_kpc
            self.ranl[i] = self.StromgrenSphere(self.data[dd]['time'], T0 = T0) / cm_per_kpc
                
    def PlotIonizationFrontEvolution(self, mp = None, anl = True, T0 = None, 
        color = 'k', ls = '--', label = None, plot_error = True, plot_solution = True):
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
        
        if plot_solution:
            self.mp.grid[0].plot(self.t / self.trec, self.rIF, 
                color = color, ls = ls)
            self.mp.grid[0].set_xlim(0, max(self.t / self.trec))
            self.mp.grid[0].set_ylim(0, 1.1 * max(max(self.rIF), max(self.ranl)))
            self.mp.grid[0].set_ylabel(r'$r \ (\mathrm{kpc})$') 
        
        if plot_error:     
            self.mp.grid[1].plot(self.t / self.trec, self.rIF / self.ranl, 
                color = color, ls = ls, label = label)
            self.mp.grid[1].set_xlim(0, max(self.t / self.trec))
            self.mp.grid[1].set_ylim(0.94, 1.05)
            self.mp.grid[1].set_xlabel(r'$t / t_{\mathrm{rec}}$')
            self.mp.grid[1].set_ylabel(r'$r_{\mathrm{num}} / r_{\mathrm{anl}}$') 
            self.mp.grid[0].xaxis.set_ticks(np.linspace(0, 4, 5))
            self.mp.grid[1].xaxis.set_ticks(np.linspace(0, 4, 5))
        
        if mp is None: 
            self.mp.fix_ticks()      
        else:
            pl.draw()
            
    def IonizationProfile(self, species = 'H', t = [1, 10, 100], color = 'k', 
        annotate = False, xscale = 'linear', yscale = 'log', ax = None):
        """
        Plot radial profiles of species fraction (for H or He) at times t (Myr).
        """      
        
        if ax is None:
            ax = pl.subplot(111)

        ax.set_xscale('log')
        ax.set_yscale('log')
        
        if species == 'H':
            fields = ['h_1', 'h_2']
            labels = [r'$x_{\mathrm{HI}}$', r'$x_{\mathrm{HII}}$']
        
        line_num = 0
        for dd in self.data.keys():
            if self.data[dd]['time'] / self.pf['time_units'] not in t: 
                continue
            
            if line_num > 0:
                labels = [None] * len(labels)
            
            for i, field in enumerate(fields):
                ax.semilogy(self.grid.r_mid / cm_per_kpc,
                    self.data[dd][field], ls = linestyles[i], 
                    color = color, label = labels[i])

            line_num += 1
                    
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)    
        ax.set_xlabel(r'$r \ (\mathrm{kpc})$') 
        ax.set_ylabel(r'Species Fraction')  
        ax.set_ylim(1e-5, 1.5)
        
        if annotate:
            ax.legend(loc = 'lower right', ncol = len(fields), 
                frameon = False)

        pl.draw()   
        
        return ax     
            
    def TemperatureProfile(self, t = [10, 30, 100], color = 'k', ls = None, xscale = 'linear', 
        legend = True, ax = None):
        """
        Plot radial profiles of temperature at times t (Myr).
        """  
        
        if ax is None:
            ax = pl.subplot(111)
        else:
            legend = False
                
        if ls is None:
            ls = linestyles
        else:
            ls = [ls] * len(t)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        line_num = 0
        for dd in self.data.keys():
            if self.data[dd]['time'] / self.pf['time_units'] not in t: 
                continue
            
            ax.loglog(self.grid.r_mid / cm_per_kpc,
                self.data[dd]['T'], ls = ls[line_num], color = color, label = r'$T_K$')
                
            line_num += 1    
                
        #if self.pf['LymanAlphaContinuum'] or self.pf['LymanAlphaInjection']:
        #    self.ax.loglog(r, self.data[dd].Ts, color = color, ls = '--', label = r'$T_S$') 
        #    
        #if self.pf['CosmologicalExpansion']:
        #    self.ax.loglog([min(r), max(r)], [self.pf['CMBTemperatureNow'] * (1. + self.data[dd].z)] * 2,
        #        color = 'k', ls = ':', label = r'$T_{\gamma}$')         
        #    
        #    if legend:
        #        self.ax.legend(loc = 'upper right', frameon = False)
            
        ax.set_xscale(xscale)
        ax.set_xlabel(r'$r \ (\mathrm{kpc})$')
        ax.set_ylabel(r'Temperature $(K)$')
        pl.draw()
        
        return ax                

class AnalyzeOld:
    def __init__(self, pf, retabulate = False):
        self.ds = Dataset(pf)
        self.data = self.ds.data
        self.t = self.ds.t
        self.dt = self.ds.dt
        
        self.pf = self.ds.pf        # dict
        self.g = InitializeGrid(self.pf)   
        self.cosm = Cosmology(self.pf)
        self.rs = RadiationSources(self.pf, retabulate = retabulate, path = self.ds.path_to_output)
        self.esec = SecondaryElectrons(self.pf)         
        
        # Convenience
        self.GridDimensions = int(self.pf['GridDimensions'])
        self.grid = np.arange(self.GridDimensions)
                
        # Deal with log-grid
        if self.pf['LogarithmicGrid']:
            self.r = np.logspace(np.log10(self.pf['StartRadius'] * self.pf['LengthUnits']),
                np.log10(self.pf['LengthUnits']), self.GridDimensions + 1)
        else:
            self.r = np.linspace(self.pf['StartRadius'] * self.pf['LengthUnits'], 
                self.pf['LengthUnits'], self.GridDimensions + 1)
        
        self.dx = np.diff(self.r)   
        self.r = self.r[0:-1]
                
        self.Vsh = 4. * np.pi * ((self.r + self.dx)**3 - self.r**3) / 3. / cm_per_mpc**3
                
        # Shift radii to cell-centered values
        self.r += self.dx / 2.   
                                
        # Store bins used for PDFs/CDFs
        self.bins = {}
          
        # Inspect instance    
        #self.inspect = Inspect(self)    
        
    def StromgrenSphere(self, t, sol = 0, T0 = None, helium_correction = 0):
        """
        Classical analytic solution for expansion of an HII region in an isothermal medium.  Given the time
        in seconds, will return the I-front radius in centimeters.
        
        Future: sol = 1 will be the better "analytic" solution.
        """
        
        # Stuff for analytic solution
        if sol == 0:
            if T0 is not None: 
                T = T0
            else: 
                T = self.data[0].T[0]
                
            AHe = 1
            if helium_correction:
                AHe = 1. / (1. - 3. * self.cosm.Y / 4.)
            nH = AHe * self.pf['DensityUnits'] * (1. - self.cosm.Y) / m_H
            self.Ndot = self.pf['SpectrumPhotonLuminosity']
            self.alpha_HII = 2.6e-13 * (T / 1.e4)**-0.85
            self.trec = 1. / self.alpha_HII / self.data[0].n_HI[-1]                                         # s
            self.rs = (3. * self.Ndot / 4. / np.pi / self.alpha_HII / nH**2)**(1. / 3.)  # cm
        
        return self.rs * (1. - np.exp(-t / self.trec))**(1. / 3.) + self.pf['StartRadius']
        
    def LocateIonizationFront(self, dd, species = 0):
        """
        Find the position of the ionization front in data dump 'dd'.
        """
        
        if species == 0:
            return np.interp(0.5, self.data[dd].x_HI, self.data[dd].r)
        else:
            return np.interp(0.5, self.data[dd].x_HeI, self.data[dd].r)
        
    def ComputeIonizationFrontEvolution(self, T0 = None, helium_correction = 0):
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
            self.ranl[i] = self.StromgrenSphere(self.data[dd].t, T0 = T0, 
                helium_correction = helium_correction) / cm_per_kpc
                
    def PlotIonizationFrontEvolution(self, mp = None, anl = True, T0 = None, helium_correction = 0,
        color = 'k', ls = '--'):
        """
        Compute analytic and numerical I-front radii vs. time and plot.
        """    

        self.ComputeIonizationFrontEvolution(T0 = T0, helium_correction = helium_correction)

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
     
    def TemperatureProfile(self, t = 10, color = 'k', ls = '-', xscale = 'linear', legend = True):
        """
        Plot radial profiles of temperature at times t (Myr).
        """  
        
        if not hasattr(self, 'ax'):
            self.ax = pl.subplot(111)
        else:
            if legend:
                legend = False
        
        self.ax.set_xscale('log')
        self.ax.set_yscale('log')
        
        for dd in self.data.keys():
            if self.data[dd].t / self.pf['TimeUnits'] != t: 
                continue
            
            break
            
        r = self.data[dd].r / self.pf['LengthUnits']
        self.ax.loglog(r, self.data[dd].T, ls = ls, color = color, label = r'$T_K$')
                
        if self.pf['LymanAlphaContinuum'] or self.pf['LymanAlphaInjection']:
            self.ax.loglog(r, self.data[dd].Ts, color = color, ls = '--', label = r'$T_S$') 
            
        if self.pf['CosmologicalExpansion']:
            self.ax.loglog([min(r), max(r)], [self.pf['CMBTemperatureNow'] * (1. + self.data[dd].z)] * 2,
                color = 'k', ls = ':', label = r'$T_{\gamma}$')         
            
            if legend:
                self.ax.legend(loc = 'upper right', frameon = False)
            
        self.ax.set_xscale(xscale)    
        self.ax.set_xlabel(r'$r / L_{\mathrm{box}}$') 
        self.ax.set_ylabel(r'Temperature $(K)$')  
        pl.draw()        
        
    def BrightnessTemperatureProfile(self, t, color = 'k', ls = '-'):
        """
        dTb(r)
        """   
        
        self.ax = pl.subplot(111)
        
        for dd in self.data.keys():
            if self.data[dd].t / self.pf['TimeUnits'] != t: 
                continue
            
            break
                    
        self.ax.plot(self.data[dd].r / cm_per_kpc, self.data[dd].dTb, color = color, ls = ls)
        
        self.ax.set_xlabel(r'$r \ (\mathrm{kpc})$')
        self.ax.set_ylabel(r'$\delta T_b \ (\mathrm{mK})$')
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
                exec('self.ax.semilogy(self.data[%i].r / cm_per_kpc, \
                    self.data[%i].%s, ls = \'%s\', color = \'%s\', label = r\'$x_{\mathrm{HI}}$\')' % (dd, dd, 'x_HI', '-', color))
                exec('self.ax.semilogy(self.data[%i].r / cm_per_kpc, \
                    self.data[%i].%s, ls = \'%s\', color = \'%s\', label = r\'$x_{\mathrm{HII}}$\')' % (dd, dd, 'x_HII', '--', color))
            if species == 'He':
                exec('self.ax.semilogy(self.data[%i].r / cm_per_kpc, \
                    self.data[%i].%s, ls = \'%s\', color = \'%s\', label = r\'$x_{\mathrm{HeI}}$\')' % (dd, dd, 'x_HeI', '-', color))
                exec('self.ax.semilogy(self.data[%i].r / cm_per_kpc, \
                    self.data[%i].%s, ls = \'%s\', color = \'%s\', label = r\'$x_{\mathrm{HeII}}$\')' % (dd, dd, 'x_HeII', '--', color))
                exec('self.ax.semilogy(self.data[%i].r / cm_per_kpc, \
                    self.data[%i].%s, ls = \'%s\', color = \'%s\', label = r\'$x_{\mathrm{HeIII}}$\')' % (dd, dd, 'x_HeIII', ':', color))                
                        
        self.ax.set_xscale(xscale)
        self.ax.set_yscale(yscale)    
        self.ax.set_xlabel(r'$r \ (\mathrm{kpc})$') 
        self.ax.set_ylabel(r'Species Fraction')  
        
        if annotate:
            if species == 'H':
                self.ax.legend(loc = 'lower right', ncol = 2, frameon = False)
            if species == 'He':
                self.ax.legend(loc = 'lower right', ncol = 3, frameon = False)    
        
        pl.draw()
        
    def RadialProfileMovie(self, field = 'x_HI', out = 'frames', xscale = 'linear',
        title = True, t_start = 0):
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
            
            if self.data[dd].t / self.pf['TimeUnits'] < t_start:
                continue
            
            exec('ax.loglog(self.data[%i].r / self.pf[\'LengthUnits\'], \
                self.data[%i].%s, ls = \'-\', color = \'k\')' % (dd, dd, field))

            ax.set_xlim(self.data[0].r[0] / self.pf['LengthUnits'], 1)        
            ax.set_ylim(mi, ma)    
            ax.set_xscale(xscale) 
            
            if title:
                ax.set_title(r'$t = %g \ \mathrm{Myr}$' % (self.data[dd].t / self.pf['TimeUnits']))
                    
            pl.savefig('%s/dd%s_%s.png' % (out, str(dd).zfill(4), field))                        
            ax.clear()
            
        pl.close()    
        
    def Ionization_Temperature_Movie(self, out = 'frames', xscale = 'linear', title = True):
        """
        Meant to answer Coughlitron's question about outside-in recombination.
        """    
                
        for dd in self.data.keys():
            
            mp = multiplot(dims = (2, 1), useAxesGrid = False)
            
            exec('mp.grid[0].semilogy(self.data[%i].r / self.pf[\'LengthUnits\'], \
                self.data[%i].x_HI, ls = \'-\', color = \'k\', label = r\'$x_{\mathrm{HI}}$\')' % (dd, dd))
            exec('mp.grid[0].semilogy(self.data[%i].r / self.pf[\'LengthUnits\'], \
                self.data[%i].x_HII, ls = \'--\', color = \'k\', label = r\'$x_{\mathrm{HII}}$\')' % (dd, dd))    
            exec('mp.grid[1].semilogy(self.data[%i].r / self.pf[\'LengthUnits\'], \
                self.data[%i].T, ls = \'-\', color = \'k\')' % (dd, dd))
                
            mp.grid[0].set_xlim(self.data[0].r[0] / self.pf['LengthUnits'], 1)        
            mp.grid[1].set_xlim(self.data[0].r[0] / self.pf['LengthUnits'], 1)        
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
            if self.data[dd].t / self.pf['TimeUnits'] not in t: 
                continue
                
            this_t = int(self.data[dd].t / self.pf['TimeUnits'])
        
            self.mp.grid[0].semilogy(self.data[dd].r / self.pf['LengthUnits'], self.data[dd].x_HI, color = color, ls = ls[ct], 
                label = r'$t = %i \ \mathrm{Myr}$' % this_t)
            self.mp.grid[1].semilogy(self.data[dd].r / self.pf['LengthUnits'], self.data[dd].T, color = color, ls = ls[ct])
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
        Return time evolution of a given quantity in one cell.
        """    
        
        z = []
        time = []
        value = []
        for dd in self.data.keys():
            if self.pf['CosmologicalExpansion']:
                z.append(self.data[dd].z)
            else:
                z.append(None)

            time.append(self.data[dd].t)
            exec('value.append(self.data[%i].%s[%i])' % (dd, field, cell))
        
        return np.array(time), np.array(z), np.array(value)
        
    def CellTimeSeries(self, cell = 0, species = 0, field = 'x_HI', color = 'k', ls = '-'):
        """
        Plot cell evolution.
        """    
        
        t, z, val = self.CellTimeEvolution(cell = cell, field = field)
        
        if len(val.shape) > 1:
            val = zip(*val)[species]
            
        if field in ['Gamma', 'gamma']:
            t, z, nabs = self.CellTimeEvolution(cell = cell, field = 'nabs')
            val *= zip(*nabs)[species]
        elif field in ['Beta']:
            t, z, nabs = self.CellTimeEvolution(cell = cell, field = 'nabs')
            t, z, ne = self.CellTimeEvolution(cell = cell, field = 'n_e')
            val *= zip(*nabs)[species] * ne
        elif field in ['zeta', 'psi']:
            t, z, nabs = self.CellTimeEvolution(cell = cell, field = 'nabs')
            t, z, ne = self.CellTimeEvolution(cell = cell, field = 'n_e')
            val *= zip(*nabs)[species] * ne
        elif field in ['eta']:
            t, z, nion = self.CellTimeEvolution(cell = cell, field = 'nion')
            t, z, ne = self.CellTimeEvolution(cell = cell, field = 'n_e')
            val *= zip(*nion)[species] * ne
            
        self.ax = pl.subplot(111)
        self.ax.loglog(t / s_per_yr, val, color = color, ls = ls)  
        
        pl.draw()       
        
    def IonizationRate(self, t = 1, species = 0, color = 'k', ls = '-', legend = True, 
        plot_recomb = False, total_only = False, src = 0):
        """
        Plot total ionization rate, and lines for primary, secondary, and collisional.
        """ 
        
        if legend and hasattr(self, 'ax'):
            legend = False
        
        for dd in self.data.keys():
            if self.data[dd].t / self.pf['TimeUnits'] != t: 
                continue
                            
            Gamma = self.data[dd].Gamma[:,species,src] * self.data[dd].nabs[species,:]
            gamma = np.sum(self.data[dd].gamma[:,species,:,src] * np.transpose(self.data[dd].nabs), axis = 1)
            Beta = self.data[dd].Beta[:,species] * self.data[dd].nabs[species,:] * self.data[dd].n_e            
            ion = Gamma + Beta + gamma
            
            alpha = self.data[dd].alpha[:,species] * self.data[dd].nion[species,:] * self.data[dd].n_e
            xi = self.data[dd].xi[:,species] * self.data[dd].nion[species,:] * self.data[dd].n_e
            recomb = alpha + xi    
                
        self.ax = pl.subplot(111)
        self.ax.loglog(self.data[dd].r / self.pf['LengthUnits'], ion, color = color, ls = ls, label = 'Total')  
        if not total_only:      
            self.ax.loglog(self.data[dd].r / cm_per_kpc, Gamma, color = color, ls = '--', label = r'$\Gamma$')
            self.ax.loglog(self.data[dd].r / cm_per_kpc, gamma, color = color, ls = ':', label = r'$\gamma$')
            self.ax.loglog(self.data[dd].r / cm_per_kpc, Beta, color = color, ls = '-.', label = r'$\beta$')
                
        if plot_recomb:
            self.ax.loglog(self.data[dd].r / cm_per_kpc, recomb, color = 'b', ls = '-', label = 'Recomb.')
            self.ax.loglog(self.data[dd].r / cm_per_kpc, alpha, color = 'b', ls = '--', label = r'$\alpha$')
            self.ax.loglog(self.data[dd].r / cm_per_kpc, xi, color = 'b', ls = ':', label = r'$\xi$')
        
        self.ax.set_xlabel(r'$r \ (\mathrm{kpc})$') 
        self.ax.set_ylabel(r'Ionization Rate $(\mathrm{s}^{-1})$')
        self.ax.set_ylim(0.01 * 10**np.floor(np.log10(np.min(ion))), 10**np.ceil(np.log10(np.max(ion))))
        
        if legend:
            self.ax.legend(frameon = False, ncol = 2)
        
        pl.draw()    
        
    def HeatingRate(self, t = 1, color = 'r', ls = '-', legend = True, src = 0, 
        plot_cooling = False, label = None):
        """
        Plot total heating rate, and lines for primary, secondary, and collisional.
        """ 
        
        if legend and hasattr(self, 'ax'):
            legend = False
        
        for dd in self.data.keys():
            if self.data[dd].t / self.pf['TimeUnits'] != t: 
                continue
            
            x_HII = self.data[dd].x_HII
            fheat = np.zeros(self.GridDimensions)
            for i in xrange(self.GridDimensions):
                fheat[i] = self.esec.DepositionFraction(None, x_HII[i], 0)   
                
            tranabs = np.transpose(self.data[dd].nabs)             
            
            heat = fheat * np.sum(self.data[dd].k_H[...,src] * tranabs, axis = 1)
            zeta = np.sum(self.data[dd].zeta * tranabs, axis = 1) * self.data[dd].n_e # collisional ionization
            eta = np.sum(self.data[dd].eta * tranabs, axis = 1) * self.data[dd].n_e  # recombination
            psi = np.sum(self.data[dd].psi * tranabs, axis = 1) * self.data[dd].n_e  # collisional excitation
            omega = np.sum(self.data[dd].omega * tranabs, axis = 1) * self.data[dd].n_e # dielectric
            cool = (zeta + eta + psi + omega)
            
            if self.pf['CosmologicalExpansion']:
                cool += self.data[dd].hubble * 3. * self.data[dd].T * k_B * self.data[dd].n_B

        mi = min(np.min(heat), np.min(cool))    
        ma = max(np.max(heat), np.max(cool))    
            
        if label is None:
            heat_label = r'$\mathcal{H}_{\mathrm{tot}}$'    
        else:
            heat_label = label    
            
        self.ax = pl.subplot(111)
        self.ax.loglog(self.data[dd].r / cm_per_kpc, heat, color = color, ls = ls, label = heat_label)
        
        if plot_cooling:
            self.ax.loglog(self.data[dd].r / cm_per_kpc, cool, color = 'b', ls = '-', label = r'$\mathcal{C}_{\mathrm{tot}}$')
            self.ax.loglog(self.data[dd].r / cm_per_kpc, zeta, color = 'g', ls = '--', label = r'$\zeta$')
            self.ax.loglog(self.data[dd].r / cm_per_kpc, psi, color = 'g', ls = ':', label = r'$\psi$')
            self.ax.loglog(self.data[dd].r / cm_per_kpc, eta, color = 'c', ls = '--', label = r'$\eta$')
        
            if self.pf['MultiSpecies']:
                self.ax.loglog(self.data[dd].r / cm_per_kpc, omega, color = 'c', ls = ':', label = r'$\omega_{\mathrm{HeII}}$')
                    
            if self.pf['CosmologicalExpansion']:
                self.ax.loglog(self.data[dd].r / cm_per_kpc, self.data[dd].hubble * 3. * self.data[dd].T * k_B * self.data[dd].n_B, 
                    color = 'm', ls = '--', label = r'$H(z)$')
                
        if plot_cooling:
            ax_label = r'Heating & Cooling Rate $(\mathrm{erg/s/cm^3})$'        
        else:    
            ax_label = r'Heating Rate $(\mathrm{erg/s/cm^3})$'        
                
        self.ax.set_xlabel(r'$r \ (\mathrm{kpc})$') 
        self.ax.set_ylabel(ax_label)
        self.ax.set_ylim(0.001 * 10**np.floor(np.log10(mi)), 10**np.ceil(np.log10(ma)))
        
        if legend:
            self.ax.legend(frameon = False, ncol = 3, loc = 'lower right')
        
        pl.draw()    
        
        # Save heating and cooling rates
        self.heat = heat
        self.cool = cool
        
    def LyAFlux(self, t = 1, color = 'k', src = 0, legend = True):
        """
        Plot Lyman-alpha flux from the continuum and 'injected' photons.
        """    
        
        for dd in self.data.keys():
            if self.data[dd].t / self.pf['TimeUnits'] != t: 
                continue
                
            break
            
        if legend and hasattr(self, 'ax'):
            legend = False    
            
        r = self.data[dd].r / cm_per_kpc  
                    
        self.ax = pl.subplot(111)
        if self.pf['LymanAlphaContinuum']:
            self.ax.semilogy(r, self.data[dd].Jc[...,src], color = color, ls = '-', label = r'$J_c$')        
        
        if self.pf['LymanAlphaInjection']:
            Ji_HI = np.array(zip(*self.data[dd].Ji[...,src])[0])
            self.ax.semilogy(r, Ji_HI, color = color, ls = '--', label = r'$J_{i,\mathrm{HI}}$')        
        
            if self.pf['MultiSpecies']:
                Ji_HeI = np.array(zip(*self.data[dd].Ji[...,src])[1])
                Ji_HeII = np.array(zip(*self.data[dd].Ji[...,src])[2])
                self.ax.semilogy(r, Ji_HeI, color = color, ls = ':', label = r'$J_{i,\mathrm{HeI}}$')
                self.ax.semilogy(r, Ji_HeII, color = color, ls = '-.', label = r'$J_{i,\mathrm{HeII}}$')
            
        self.ax.semilogy([0, 1], [self.J0(self.data[dd].z)] * 2, color = 'b', ls = ':')    
        
        self.ax.set_xlabel(r'$r \ (\mathrm{kpc})$') 
        self.ax.set_ylabel(r'$J_{\alpha} \ (\mathrm{erg \ s^{-1} \ cm^{-2} \ sr^{-1} \ Hz^{-1}})$')
        
        if legend:
            self.ax.legend(frameon = False, loc = 'upper right')
        pl.draw()

    def J0(self, z):
        """
        Lyman-alpha flux corresponding to 1 photon per hydrogen atom.
        """
        return h * nu_alpha * c * self.cosm.nH0 * (1. + z)**3 / 4. / np.pi / nu_alpha
        