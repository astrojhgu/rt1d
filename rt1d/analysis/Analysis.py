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
from Multiplot import *
import rt1d.mods as rtm
from rt1d.mods.Constants import *
from rt1d.analysis.Dataset import Dataset
from rt1d.mods.ComputeCrossSections import PhotoIonizationCrossSection
from rt1d.mods.InitializeIntegralTables import InitializeIntegralTables
from rt1d.mods.Interpolate import Interpolate

class Analyze:
    def __init__(self, pf, retabulate = False):
        self.ds = Dataset(pf)
        self.data = self.ds.data
        self.pf = self.ds.pf    # dictionary
        self.g = rtm.InitializeGrid(self.pf)   
        self.iits = rtm.InitializeIntegralTables(self.pf, self.data[0], self.g)      
        self.esec = rtm.SecondaryElectrons(self.pf)  
        
        # Convenience
        self.GridDimensions = int(self.pf['GridDimensions'])
        self.grid = np.arange(self.GridDimensions)
        self.LengthUnits = self.pf['LengthUnits']
        self.StartRadius = self.pf['StartRadius']
        self.MultiSpecies = self.pf["MultiSpecies"]
                
        # Deal with log-grid
        if self.pf["LogarithmicGrid"]:
            self.r = np.logspace(np.log10(self.StartRadius * self.LengthUnits), \
                np.log10(self.LengthUnits), self.GridDimensions + 1)
        else:
            rmin = self.StartRadius * self.LengthUnits
            self.r = np.linspace(rmin, self.LengthUnits, self.GridDimensions + 1)
        
        self.dx = np.diff(self.r)   
        self.r = self.r[0:-1]
                
        self.Vsh = 4. * np.pi * ((self.r + self.dx)**3 - self.r**3) / 3. / cm_per_mpc**3        
                
        # Shift radii to cell-centered values
        self.r += self.dx / 2.   
                                
        # Store bins used for PDFs/CDFs
        self.bins = {}
        
        # Read integral table if it exists.
        self.tname = self.iits.DetermineTableName()
        if os.path.exists('%s' % self.tname) and retabulate:
            self.itabs = self.iits.TabulateRateIntegrals()
            self.interp = Interpolate(self.pf, [self.iits.HIColumn, self.iits.HeIColumn, self.iits.HeIIColumn], 
                self.itabs)
        else:
            self.itabs = self.interp = None    
        
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
        self.mp.grid[1].set_ylabel(r'$r/r_{\mathrm{anl}}$') 
        self.mp.grid[0].xaxis.set_ticks(np.linspace(0, 4, 5))
        self.mp.grid[1].xaxis.set_ticks(np.linspace(0, 4, 5))
        
        if mp is None: 
            self.mp.fix_ticks()  
     
    def TemperatureProfile(self, t = 10, color = 'k', ls = '-'):
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
                self.data[%i].T, ls = \'%s\', color = \'%s\')' % (dd, dd, '-', color))                
            
        self.ax.set_xlabel(r'$r / L_{\mathrm{box}}$') 
        self.ax.set_ylabel(r'Temperature $(K)$')  
        pl.draw()        
        
    def IonizationProfile(self, species = 'H', t = [1, 10, 100], color = 'k'):
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
                exec('self.ax.loglog(self.data[%i].r / self.pf[\'LengthUnits\'], \
                    self.data[%i].%s, ls = \'%s\', color = \'%s\')' % (dd, dd, 'x_HI', '-', color))
                exec('self.ax.loglog(self.data[%i].r / self.pf[\'LengthUnits\'], \
                    self.data[%i].%s, ls = \'%s\', color = \'%s\')' % (dd, dd, 'x_HII', '--', color))    
            if species == 'He':
                exec('self.ax.loglog(self.data[%i].r / self.pf[\'LengthUnits\'], \
                    self.data[%i].%s, ls = \'%s\', color = \'%s\')' % (dd, dd, 'x_HeI', '-', color))
                exec('self.ax.loglog(self.data[%i].r / self.pf[\'LengthUnits\'], \
                    self.data[%i].%s, ls = \'%s\', color = \'%s\')' % (dd, dd, 'x_HeII', '--', color))
                exec('self.ax.loglog(self.data[%i].r / self.pf[\'LengthUnits\'], \
                    self.data[%i].%s, ls = \'%s\', color = \'%s\')' % (dd, dd, 'x_HeIII', ':', color))                
            
        self.ax.set_xlabel(r'$r / L_{\mathrm{box}}$') 
        self.ax.set_ylabel(r'Species Fraction')  
        pl.draw()
        
    def RadialProfileMovie(self, species = 'H', out = 'frames'):
        """
        Save time-series images of 'field' to 'out' directory.
        """    
        
        if out is None:
            out = './'
        elif not os.path.exists(out):
            os.mkdir(out)

        mi, ma = (1e-5, 1.5)
        ax = pl.subplot(111)    
        ax.set_xscale('log')        
        ax.set_yscale('log')  
         
        for dd in self.data.keys():
            
            if species == 'H':
                exec('ax.loglog(self.data[%i].r / self.pf[\'LengthUnits\'], \
                    self.data[%i].x_HI, ls = \'-\', color = \'k\')' % (dd, dd))
                exec('ax.loglog(self.data[%i].r / self.pf[\'LengthUnits\'], \
                    self.data[%i].x_HII, ls = \'--\', color = \'k\')' % (dd, dd))
            else:
                exec('ax.loglog(self.data[%i].r / self.pf[\'LengthUnits\'], \
                    self.data[%i].x_HeI, ls = \'-\', color = \'k\')' % (dd, dd))
                exec('ax.loglog(self.data[%i].r / self.pf[\'LengthUnits\'], \
                    self.data[%i].x_HeII, ls = \'--\', color = \'k\')' % (dd, dd))   
                exec('ax.loglog(self.data[%i].r / self.pf[\'LengthUnits\'], \
                    self.data[%i].x_HeIII, ls = \':\', color = \'k\')' % (dd, dd))            

            ax.set_xlim(self.data[0].r[0] / self.pf['LengthUnits'], 1)        
            ax.set_ylim(mi, ma)         
                    
            pl.savefig('%s/dd%s_x%s.png' % (out, str(dd).zfill(4), species))                        
            ax.clear()
            
        pl.close()    
        
    def InspectIntegralTable(self, integral = 0, species = 0, nHI = None, nHeI = 0.0, nHeII = 0.0,
        color = 'k', ls = '-', annotate = True):
        """
        Plot integral values...or something.
        """ 
        
        if species == 0:
            s = 'HI'
        elif species == 1:
            s = 'HeI'
        else:
            s = 'HeII'
        
        # Convert integral int to string
        if integral == 0:
            integral = 'PhotoIonizationRate%i' % species 
            ylabel = r'$\Phi_{\mathrm{%s}}$' % s
        elif integral == 1:
            integral = 'ElectronHeatingRate%i' % species
            ylabel = r'$\Psi_{\mathrm{%s}}$' % s
        else:
            integral = 'TotalOpticalDepth' 
            ylabel = r'$\sum_i\int_{\nu} \tau_{i,\nu} d\nu$'
            pl.rcParams['figure.subplot.left'] = 0.2
        
        # Figure out axes
        if nHI is None:
            x = self.itabs['HIColumnValues_x']
            if self.MultiSpecies:
                i1 = np.argmin(np.abs(self.itabs['HeIColumnValues_y'] - nHeI))
                i2 = np.argmin(np.abs(self.itabs['HeIIColumnValues_z'] - nHeII))
                y = self.itabs[integral][0:,i1,i2]
            else:
                y = self.itabs[integral]
            xlabel = r'$n_{\mathrm{HI}} \ (\mathrm{cm^{-2}})$'
        elif nHeI is None:
            x = self.itabs['HeIColumnValues_y']
            i1 = np.argmin(np.abs(self.itabs['HIColumnValues_x'] - nHI))
            i2 = np.argmin(np.abs(self.itabs['HeIIColumnValues_z'] - nHeII))
            y = self.itabs[integral][i1,0:,i2]
            xlabel = r'$n_{\mathrm{HeI}} \ (\mathrm{cm^{-2}})$'
        else:
            x = self.itabs['HeIIColumnValues_z']
            i1 = np.argmin(np.abs(self.itabs['HIColumnValues_x'] - nHI))
            i2 = np.argmin(np.abs(self.itabs['HeIColumnValues_y'] - nHeI))
            y = self.itabs[integral][i1,i2,0:]
            xlabel = r'$n_{\mathrm{HeII}} \ (\mathrm{cm^{-2}})$'
            
        if not hasattr(self, 'ax'):
            self.ax = pl.subplot(111)
            
        self.ax.loglog(10**x, 10**y, color = color, ls = '-')
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        
        pl.draw()
        
    def InspectInterpolation(self, integral = 0, species = 0, nHI = np.linspace(12, 24, 1000), 
        nHeI = 0.0, nHeII = 0.0, color = 'b', ls = '-'):
        """
        Now check how good our interpolation is...
        
        Give columns in log-space.
        """   
        
        if species == 0:
            s = 'HI'
        elif species == 1:
            s = 'HeI'
        else:
            s = 'HeII'
        
        # Convert integral int to string
        if integral == 0:
            integral = 'PhotoIonizationRate%i' % species 
            ylabel = r'$\Phi_{\mathrm{%s}}$' % s
        elif integral == 1:
            integral = 'ElectronHeatingRate%i' % species
            ylabel = r'$\Psi_{\mathrm{%s}}$' % s
        else:
            integral = 'TotalOpticalDepth%i' % species 
            ylabel = r'$\sum_i\int_{\nu} \tau_{i,\nu} d\nu$'
            pl.rcParams['figure.subplot.left'] = 0.2
        
        result = []
        if type(nHI) not in [int, float]:
            x = nHI
            for col in nHI:
                tmp = [col, nHeI, nHeII]
                                
                if self.MultiSpecies:
                    indices = self.interp.GetIndices3D(tmp)
                else:
                    indices = None
                    
                result.append(self.interp.interp(indices, 
                    '%s' % integral, tmp))
            
        elif type(nHeI) not in [int, float]:
            x = nHeI
            for col in nHeI:
                tmp = [nHI, col, nHeII]
                result.append(self.interp.interp(self.interp.GetIndices3D(tmp), 
                    '%s' % integral))    
        
        else:
            x = nHeII
            for col in nHeII:
                tmp = [nHI, nHeI, col]
                result.append(self.interp.interp(self.interp.GetIndices3D(tmp), 
                    '%s' % integral))            
            
        self.ax = pl.subplot(111)
        self.ax.loglog(10**np.array(x), result, color = color, ls = ls)
                                
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
            if self.data[dd].t / self.pf['TimeUnits'] != t: 
                continue
            
            exec('self.ax.loglog(self.data[%i].r / self.pf[\'LengthUnits\'], \
                self.data[%i].%s[%i], ls = \'%s\', color = \'%s\')' % (dd, dd, coeff, species, ls, color))                
            
        self.ax.set_xlabel(r'$r / L_{\mathrm{box}}$') 
        self.ax.set_ylabel(ylabel)         
                                
        pl.draw()      
        
    def InspectIonizationRate(self, t = 1, species = 0, color = 'k', legend = True):
        """
        Plot total ionization rate, and lines for primary, secondary, and collisional.
        """ 
        
        if legend and hasattr(self, 'ax'):
            legend = False
        
        for dd in self.data.keys():
            if self.data[dd].t / self.pf['TimeUnits'] != t: 
                continue
            
            Gamma = self.data[dd].Gamma[species]
            gamma = self.data[dd].gamma[species]
            Beta = self.data[dd].Beta[species]
            tot = Gamma + gamma + Beta
                
        self.ax = pl.subplot(111)
        self.ax.loglog(self.data[dd].r / self.pf['LengthUnits'], tot, color = color, ls = '-', label = 'Total')        
        self.ax.loglog(self.data[dd].r / self.pf['LengthUnits'], Gamma, color = color, ls = '--', label = r'$\Gamma$')
        self.ax.loglog(self.data[dd].r / self.pf['LengthUnits'], gamma, color = color, ls = ':', label = r'$\gamma$')
        self.ax.loglog(self.data[dd].r / self.pf['LengthUnits'], Beta, color = color, ls = '-.', label = r'$\beta$')
        
        self.ax.set_xlabel(r'$r / L_{\mathrm{box}}$') 
        self.ax.set_ylabel(r'Ionization Rate')
        self.ax.set_ylim(0.01 * 10**np.floor(np.log10(np.min(tot))), 10**np.ceil(np.log10(np.max(tot))))
        
        if legend:
            self.ax.legend(frameon = False, ncol = 2)
        
        pl.draw()
        
    def InspectHeatingRate(self, t = 1, color = 'k', legend = True):
        """
        Plot total ionization rate, and lines for primary, secondary, and collisional.
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
            
            heat = fheat * np.sum(self.data[dd].k_H * self.data[dd].nabs, axis = 0)
            zeta = np.sum(self.data[dd].zeta * self.data[dd].nabs, axis = 0) * self.data[dd].n_e # collisional ionization
            eta = np.sum(self.data[dd].eta * self.data[dd].nabs, axis = 0) * self.data[dd].n_e  # recombination
            psi = np.sum(self.data[dd].psi * self.data[dd].nabs, axis = 0) * self.data[dd].n_e  # collisional excitation
            omega = np.sum(self.data[dd].omega * self.data[dd].nabs, axis = 0) * self.data[dd].n_e # dielectric
            cool = (zeta + eta + psi + omega)
            
        mi = min(np.min(heat), np.min(cool))    
        ma = max(np.max(heat), np.max(cool))    
            
        self.ax = pl.subplot(111)
        self.ax.loglog(self.data[dd].r / self.pf['LengthUnits'], heat, color = 'r', ls = '-', label = r'$\mathcal{H}_{\mathrm{tot}}$')
        self.ax.loglog(self.data[dd].r / self.pf['LengthUnits'], cool, color = 'b', ls = '-', label = r'$\mathcal{C}_{\mathrm{tot}}$')
        self.ax.loglog(self.data[dd].r / self.pf['LengthUnits'], zeta, color = 'g', ls = '--', label = r'$\zeta$')
        self.ax.loglog(self.data[dd].r / self.pf['LengthUnits'], psi, color = 'g', ls = ':', label = r'$\psi$')
        self.ax.loglog(self.data[dd].r / self.pf['LengthUnits'], eta, color = 'c', ls = '--', label = r'$\eta$')
        self.ax.loglog(self.data[dd].r / self.pf['LengthUnits'], omega, color = 'c', ls = ':', label = r'$\omega_{\mathrm{HeII}}$')
                
        self.ax.set_xlabel(r'$r / L_{\mathrm{box}}$') 
        self.ax.set_ylabel(r'Heating/Cooling Rate')
        self.ax.set_ylim(0.001 * 10**np.floor(np.log10(mi)), 10**np.ceil(np.log10(ma)))
        
        if legend:
            self.ax.legend(frameon = False, ncol = 3, loc = 'lower right')
        
        pl.draw()    
        
        # Save heating and cooling rates
        self.heat = heat
        self.cool = cool
        
    def InspectTimestepEvolution(self, t = 1):
        """
        Plot timestep as a function of radius.
        """    
        
        self.dt = np.zeros([3, self.GridDimensions])
        for dd in self.data.keys():
            if self.data[dd].t / self.pf['TimeUnits'] != t: 
                continue
        
            dHIIdt = self.data[dd].n_HI * \
                (self.data[dd].Gamma[0] + self.data[dd].gamma[0] + self.data[dd].Beta[0] * self.data[dd].n_e) \
                - self.data[dd].n_HII * self.data[dd].n_e * self.data[dd].alpha[0]
                
            dHeIIdt = self.data[dd].n_HeI * \
                (self.data[dd].Gamma[1] + self.data[dd].gamma[1] + self.data[dd].Beta[1] * self.data[dd].n_e) \
                + self.data[dd].alpha[2] * self.data[dd].n_e * self.data[dd].n_HeIII \
                - self.data[dd].n_HeII * self.data[dd].n_e * (self.data[dd].alpha[1] + self.data[dd].Beta[2]) \
                - self.data[dd].xi[1] * self.data[dd].n_e * self.data[dd].n_HeIII
                
            dHeIIIdt = self.data[dd].n_HeII * \
                (self.data[dd].Gamma[2] + self.data[dd].gamma[2] + self.data[dd].Beta[2] * self.data[dd].n_e) \
                - self.data[dd].n_HeIII * self.data[dd].n_e * self.data[dd].alpha[2]
                
            self.dt[0] = self.pf['MaxHIIChange'] * self.data[dd].n_HI / abs(dHIIdt) / self.pf['TimeUnits']
            self.dt[1] = self.pf['MaxHeIIChange'] * self.data[dd].n_HeII / abs(dHeIIdt) / self.pf['TimeUnits']
            self.dt[2] = self.pf['MaxHeIIIChange'] * self.data[dd].n_HeIII / abs(dHeIIIdt) / self.pf['TimeUnits']
            
            break
            
        mi = 10**np.floor(np.log10(np.min(self.dt)))
        ma = 1e2
                        
        self.ax = pl.subplot(111)
        self.ax.loglog(self.data[dd].r / self.pf['LengthUnits'], self.dt[0], color = 'k', label = r'$\Delta t_{\mathrm{HII}}$')
        i1 = np.argmin(np.abs(self.data[dd].tau[0] - self.pf["OpticalDepthDefiningIFront"][0]))
        self.ax.loglog([self.data[dd].r[i1] / self.pf['LengthUnits']] * 2, [mi, ma], color = 'b', ls = '-')
        
        if self.pf["MultiSpecies"]:
            self.ax.loglog(self.data[dd].r / self.pf['LengthUnits'], self.dt[1], color = 'k', ls = '--', label = r'$\Delta t_{\mathrm{HeII}}$')
            self.ax.loglog(self.data[dd].r / self.pf['LengthUnits'], self.dt[2], color = 'k', ls = ':', label = r'$\Delta t_{\mathrm{HeIII}}$')
            i2 = np.argmin(np.abs(self.data[dd].tau[1] - self.pf["OpticalDepthDefiningIFront"][1]))
            i3 = np.argmin(np.abs(self.data[dd].tau[2] - self.pf["OpticalDepthDefiningIFront"][2]))
            self.ax.loglog([self.data[dd].r[i2] / self.pf['LengthUnits']] * 2, [mi, ma], color = 'b', ls = '--')
            self.ax.loglog([self.data[dd].r[i3] / self.pf['LengthUnits']] * 2, [mi, ma], color = 'b', ls = ':')
         
        # Minimum dtphot    
        self.ax.loglog(self.data[dd].r / self.pf['LengthUnits'], self.data[dd].dtPhoton, color = 'g', ls = '-', label = r'$\Delta t_{\mathrm{min}}$')
        
        # Timestep chosen for next step   
        self.ax.loglog([self.data[dd].r[0] / self.pf['LengthUnits'], self.data[dd].r[-1] / self.pf['LengthUnits']], 
            [np.min(self.data[dd].dtPhoton)] * 2, color = 'r', ls = '-')        
        
        self.ax.set_xlabel(r'$r / L_{\mathrm{box}}$') 
        self.ax.set_ylabel(r'$\Delta t_{\mathrm{phot}} / \mathrm{TimeUnits}$') 
        self.ax.legend(loc = 'lower right', ncol = 2, frameon = False)
        self.ax.annotate(r'$\Delta t_{\mathrm{next}}$', 
            (self.data[dd].r[self.data[dd].dtPhoton == np.min(self.data[dd].dtPhoton)] / self.pf['LengthUnits'],
             0.85 * np.min(self.data[dd].dtPhoton)),
            va = 'top', ha = 'center')
            
        self.ax.set_ylim(mi, ma)    
        
        pl.draw()    
        
    def InspectSolverSpeed(self, t = 1, color = 'k', ls = '-'):
        """
        Plot ODE step as a function of radius.
        """    
        
        for dd in self.data.keys():
            if self.data[dd].t / self.pf['TimeUnits'] != t: 
                continue
        
            self.ax = pl.subplot(111)
            self.ax.loglog(self.data[dd].r / self.pf["LengthUnits"], self.data[dd].odestep, color = color, ls = ls)
            
            self.ax.set_xlabel(r'$r / L_{\mathrm{box}}$') 
            self.ax.set_ylabel(r'$\Delta t_{\mathrm{ODE}}$') 
            
            break
            
        pl.draw()    
                    
    def ComputeDistributionFunctions(self, field, normalize = True, bins = 20, volume = False):
        """
        Histogram all fields.
        """            
                
        pdf = []
        cdf = []
        icdf = []
        for dd in self.data.keys():
            exec('hist, bin_edges = np.histogram(self.data[{0}].{1}, bins = bins)'.format(dd, field))                
                            
            if volume: hist = hist**3                
                            
            if normalize: norm = float(np.sum(hist))
            else: norm = 1.
            
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
        
        if df == 'pdf': hist = self.pdf
        else: hist = self.cdf
        
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
            for i, element in enumerate(result): result[i] = (bins[i] + bins[i + 1]) / 2.
                
        return result
        
        

    
    