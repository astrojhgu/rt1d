"""
AnalyzeSimulation.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2011-06-17.

Description: Functions to calculate various quantities from our rt1d datasets.
     
"""

import os, re
import numpy as np
import matplotlib.pyplot as pl
from math import floor, ceil
from ..run import Simulation
from ..static.Grid import Grid
from ..physics.Constants import *

try:
    import h5py
except ImportError:
    pass

try:
    from multiplot import multipanel
    multiplot = True
except ImportError:
    multiplot = False

linestyles = ['-', '--', ':', '-.']

class Simulation:
    def __init__(self, checkpoints, rs=None):
        
        if isinstance(checkpoints, Simulation):
            checkpoints = checkpoints.checkpoints
        
        # Load contents of hdf5 file
        if type(checkpoints) is str:
            import pickle
            f = h5py.File(checkpoints)
            
            self.pf = {}
            for key in f['parameters']:
                self.pf[key] = f['parameters'][key].value
            
            self.data = {}
            for key in f.keys():
                if not f[key].attrs.get('is_data'):
                    continue
                
                if key == 'parameters':
                    continue
                
                dd = key#int(key.strip('dd'))
                self.data[dd] = {}
                for element in f[key]:
                    self.data[dd][element] = f[key][element].value    
            
            f.close()
            
            self.grid = Grid(dims=self.pf['grid_cells'], 
                length_units=self.pf['length_units'], 
                start_radius=self.pf['start_radius'],
                approx_Salpha=self.pf['approx_Salpha'],
                approx_lya=self.pf['approx_lya'])
                            
            self.grid.set_ics(self.data['dd0000'])
            self.grid.set_chemistry(Z=self.pf['Z'], 
                abundance=self.pf['abundances'], 
                approx_helium=self.pf['approx_helium'])
            self.grid.set_density(self.data['dd0000']['rho'])
        
        # Read contents from CheckPoints class instance            
        else:
            self.checkpoints = checkpoints
            self.grid = checkpoints.grid
            self.pf = checkpoints.pf
            self.data = checkpoints.data
            
        self.rs = rs
        
    def EmergentSpectrum(self, t, cell=-1, norm=False, logE=False):
        """
        Compute emergent spectrum at time t (Myr) and given cell in grid. By
        default, cell=-1, i.e., compute the spectrum that emerges from the grid.
        If norm=True, do not apply geometrical dilution.
        
        Returns
        -------
        E, Fin, Fout
        
        """        
        
        if not hasattr(self, 'rs'):
            raise ValueError('RadiationSource class instance required.')
            
        if logE:
            E = np.logspace(np.log10(self.rs.Emin), np.log10(self.rs.Emax))
        else:
            E = np.linspace(self.rs.Emin, self.rs.Emax)
        
        F = np.array(map(self.rs.Spectrum, E))
        
        for dd in self.data: 
            if self.data[dd]['time'] / s_per_myr != t:
                continue
                
            N, logN, Nc = self.grid.ColumnDensity(self.data[dd])
            
            Ntot = {}
            for absorber in self.grid.absorbers:
                Ntot[absorber] = N[absorber][cell]
            
            tau = self.rs.tab.SpecificOpticalDepth(E, Ntot)
            
            break
        
        out = F * np.exp(-tau)
        if not norm:
            out *= self.rs.BolometricLuminosity(t * s_per_myr) \
                / (4. * np.pi * self.grid.r[cell]**2)
        
        return E, F, out
            
    def StromgrenSphere(self, t, sol=0, T0=None):
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
                T = self.data['dd0000']['Tk'][0]
                
            n_H = self.grid.n_H[0]
            self.Qdot = self.pf['source_qdot']
            self.alpha_HII = 2.6e-13 * (T / 1.e4)**-0.85
            self.trec = 1. / self.alpha_HII / self.data['dd0000']['h_1'][0] / n_H # s
            self.rstrom = (3. * self.Qdot \
                    / 4. / np.pi / self.alpha_HII / n_H**2)**(1. / 3.)  # cm
        else:
            raise NotImplementedError('')
        
        return self.rstrom * (1. - np.exp(-t / self.trec))**(1. / 3.) \
            + self.pf['start_radius']
        
    def LocateIonizationFront(self, dd, species = 0):
        """
        Find the position of the ionization front in data dump 'dd'.
        """
        
        if species == 0:
            return np.interp(0.5, self.data[dd]['h_1'], self.grid.r_mid)
        else:
            return np.interp(0.5, self.data[dd]['he_1'], self.grid.r_mid)
        
    def ComputeIonizationFrontEvolution(self, T0=None, xmin=0.1, xmax=0.9):
        """
        Find the position of the I-front at all times, and compute value of 
        analytic solution.
        """    
                
        # First locate I-front for all data dumps and compute analytic solution
        self.t = []
        self.rIF = []
        self.drIF, self.r1_IF, self.r2_IF = [], [], []        
        self.ranl = []
        for i, dd in enumerate(self.data.keys()):
            if dd == 'dd0000':
                continue
                 
            self.t.append(self.data[dd]['time'])
            self.rIF.append(self.LocateIonizationFront(dd) / cm_per_kpc)
            self.ranl.append(self.StromgrenSphere(self.data[dd]['time'], 
                T0=T0) / cm_per_kpc)
            
            x1 = np.interp(1.-xmax, self.data[dd]['h_1'], self.grid.r_mid)
            x2 = np.interp(1.-xmin, self.data[dd]['h_1'], self.grid.r_mid)
            self.drIF.append(x2-x1)
            self.r1_IF.append(x1)
            self.r2_IF.append(x2)
                
        order = np.argsort(self.t)
        self.t = np.array(self.t)[order]
        self.rIF = np.array(self.rIF)[order]
        self.ranl = np.array(self.ranl)[order]
        self.drIF = np.array(self.drIF)[order]        
        self.r1_IF = np.array(self.r1_IF)[order]
        self.r2_IF = np.array(self.r2_IF)[order]
                
    def PlotIonizationFrontEvolution(self, mp=None, anl=True, T0=None, 
        color='k', ls='--', label=None, plot_error=True, plot_solution=True):
        """
        Compute analytic and numerical I-front radii vs. time and plot.
        """    

        self.ComputeIonizationFrontEvolution(T0=T0)

        hadmp = False
        if mp is not None: 
            mp = mp    
            hadmp = True
        else: 
            if not multiplot:
                raise ImportError('multiplot package required by this method.')

            mp = multipanel(dims=(2, 1))

        if anl: 
            mp.grid[1].plot(self.t / self.trec, self.ranl, ls='-', color='k')
            
        if plot_solution:
            mp.grid[1].plot(self.t / self.trec, self.rIF, 
                color = color, ls = ls)
            mp.grid[1].set_xlim(0, max(self.t / self.trec))
            mp.grid[1].set_ylim(0, 1.1 * max(max(self.rIF), max(self.ranl)))
            mp.grid[1].set_ylabel(r'$r \ (\mathrm{kpc})$') 
        
        if plot_error:
            mp.grid[0].plot(self.t / self.trec, self.rIF / self.ranl, 
                color=color, ls=ls, label=label)
            mp.grid[0].set_xlim(0, max(self.t / self.trec))
            mp.grid[0].set_ylim(0.94, 1.05)
            
            if not hadmp:
                mp.grid[0].set_yticks(np.arange(0.94, 1.04, 0.02))
                mp.grid[0].set_xlabel(r'$t / t_{\mathrm{rec}}$')
                mp.grid[0].set_ylabel(r'$r_{\mathrm{num}} / r_{\mathrm{anl}}$') 
                mp.grid[1].set_xticks(np.linspace(0, 4, 5))
                mp.grid[0].set_xticks(np.linspace(0, 4, 5))
                mp.grid[1].set_xticklabels(np.linspace(0, 4, 5))
                mp.grid[0].set_xticklabels(np.linspace(0, 4, 5))
        
                mp.fix_ticks()
        
        pl.draw()
            
        return mp
            
    def IonizationProfile(self, species='h', t=[1, 10, 100], color='k', 
        annotate=False, xscale='linear', yscale='log', ax=None,
        normx=False, marker=None, s=50, facecolors=None,
        iononly=False, ls=None, label=None):
        """
        Plot radial profiles of species fraction (for H or He) at times t (Myr).
        """      
        
        if ax is None:
            ax = pl.subplot(111)

        ax.set_xscale('log')
        ax.set_yscale('log')
        
        fields = self.grid.ions_by_parent[species]
        if species == 'h':
            labels = [r'$x_{\mathrm{HI}}$', r'$x_{\mathrm{HII}}$']
        elif species == 'he':
            labels = [r'$x_{\mathrm{HeI}}$', r'$x_{\mathrm{HeII}}$', 
                r'$x_{\mathrm{HeIII}}$']
                
        if label is not None:
            labels = [label] + [None] * (len(labels) - 1)
        
        for dd in self.data.keys():
            t_time_units = self.data[dd]['time'] / self.pf['time_units']
            if t_time_units not in t: 
                continue
                
            line_num = t.index(t_time_units)
            if line_num > 0:
                lab = [None] * len(labels)
            else:
                lab = labels
            
            for i, field in enumerate(fields):
                
                if iononly and field in self.grid.ions:
                    continue
                
                if ls is None:
                    lss = linestyles[i]
                else:
                    lss = ls
                
                if marker is None:
                    ax.plot(self.grid.r_mid / cm_per_kpc,
                        self.data[dd][field], ls=lss, 
                        color=color, label=lab[i])
                else:
                    ax.scatter(self.grid.r_mid / cm_per_kpc,
                        self.data[dd][field], marker=marker, s=s, 
                        color=color, label=lab[i], facecolors=facecolors)
                    
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)    
        ax.set_xlabel(r'$r \ (\mathrm{kpc})$')
        
        if iononly:
            ax.set_ylabel(r'Ionized Fraction$')
        else:    
            ax.set_ylabel(r'Species Fraction')
        ax.set_ylim(1e-5, 1.5)
        
        if annotate:
            ax.legend(loc='best', ncol=len(fields), 
                frameon=False)

        pl.draw()   
        
        return ax     
            
    def TemperatureProfile(self, t=[10,30,100], color='k', ls=None, 
        xscale='linear', yscale='log', ax = None, normx=False, 
        marker=None, s=50, facecolors='none', label=None):
        """
        Plot radial profiles of temperature at times t (Myr).
        """  
        
        if ax is None:
            ax = pl.subplot(111)
        
        if ls is None:
            ls = linestyles
        else:
            ls = [ls] * len(t)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        for dd in self.data.keys():
            t_time_units = self.data[dd]['time'] / self.pf['time_units']
            if t_time_units not in t: 
                continue
                
            line_num = t.index(t_time_units)
            if line_num > 0:
                lab = None
            else:
                lab = label
            
            if marker is None:
                ax.plot(self.grid.r_mid / cm_per_kpc,
                    self.data[dd]['Tk'], ls=ls[line_num], color=color, 
                    label=lab)
            else:
                ax.scatter(self.grid.r_mid / cm_per_kpc,
                    self.data[dd]['Tk'], color=color, 
                    label=lab, marker=marker, s=s,
                    facecolors=facecolors)
                                        
        #if self.pf['LymanAlphaContinuum'] or self.pf['LymanAlphaInjection']:
        #    self.ax.loglog(r, self.data[dd].Ts, color = color, ls = '--', label = r'$T_S$') 
        #    
        #if self.pf['CosmologicalExpansion']:
        #    self.ax.loglog([min(r), max(r)], [self.pf['CMBTemperatureNow'] * (1. + self.data[dd].z)] * 2,
        #        color = 'k', ls = ':', label = r'$T_{\gamma}$')         
        #    
            
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlabel(r'$r \ (\mathrm{kpc})$')
        ax.set_ylabel(r'Temperature $(K)$')
        pl.draw()
        
        return ax       
        
    def BrightnessTemperatureProfile(self, t=[10, 20, 30], z=None, 
        color = 'k', ls=None, 
        xscale='linear', yscale='linear', ax = None, normx = False, 
        marker=None, s=50, facecolors='none', label=None):
        """
        Plot radial profiles of temperature at times t (Myr).
        """
        
        hadax = False if ax is None else True
        
        if ax is None:
            ax = pl.subplot(111)
        
        if ls is None:
            ls = linestyles
        else:
            ls = [ls] * len(t)
        
        for dd in self.data.keys():
            t_time_units = self.data[dd]['time'] / self.pf['time_units']
            if t_time_units not in t: 
                continue
                
            line_num = t.index(t_time_units)
            if line_num > 0:
                lab = None
            else:
                lab = label
                
            z = self.data[dd]['redshift']
            xHII = self.data[dd]['h_2']
            Ts = self.data[dd]['Ts']#self.grid.hydr.Ts(self.data[dd], z)
            dTb = self.grid.hydr.DifferentialBrightnessTemperature(z, xHII, Ts)
            
            if marker is None:
                ax.plot(self.grid.r_mid / cm_per_kpc, dTb,
                    ls=ls[line_num], color=color, 
                    label=lab)
            else:
                ax.scatter(self.grid.r_mid / cm_per_kpc, dTb,
                    color=color, 
                    label=lab, marker=marker, s=s,
                    facecolors=facecolors)
        
        if not hadax:
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            ax.set_xlabel(r'$r \ (\mathrm{kpc})$')
            ax.set_ylabel(r'$\delta T_b \ (\mathrm{mK})$')
            ax.set_ylim(dTb.min() - (dTb.min() % 5), dTb.max() + (5 - dTb.max() % 5))
        pl.draw()
        
        return ax
        
    def CellEvolution(self, cell=0, field='x_HI', redshift=False):
        """
        Return time or redshift evolution of a given quantity in given cell.
        """    
        
        time = []
        value = []
        for dd in self.data.keys():
            if field not in self.data[dd].keys():
                continue

            if redshift and 'redshift' in self.data[dd].keys():
                time.append(self.data[dd]['redshift'])
            else:
                time.append(self.data[dd]['time'])
            
            value.append(self.data[dd][field][cell])
            
        # Sort result    
        order = np.argsort(time)
        time = np.array(time)[order]
        value = np.array(value)[order]
        
        return time, value
    
    def IonizationRate(self, t=1, absorber='h_1', color='k', ls='-', 
        legend=True, plot_recomb=False, total_only=False, src=0, ax=None):
        """
        Plot ionization rate.
        
        Parameters
        ----------
        total_only : bool
            If True, will only plot total ionization rate. If False, will
            separate out into photo-ionization, secondary ionization, 
            and collisional ionization.
            
        """ 
        
        if type(t) is not list:
            t = [t]
        
        if ax is None:
            ax = pl.subplot(111)
        
        i = self.grid.absorbers.index(absorber)
        for dd in self.data.keys():
            if self.data[dd]['time'] / self.pf['time_units'] not in t: 
                continue
            
            ne = self.data[dd]['de']
            nabs = self.data[dd][absorber] * self.grid.x_to_n[absorber]
            nion = self.data[dd]['h_2'] * self.grid.x_to_n[absorber]
              
            if 'Gamma_0' in self.data[dd].keys():
                Gamma = self.data[dd]['Gamma_%i' % src][...,i] * nabs
                
                gamma = 0.0
                for j, donor in enumerate(self.grid.absorbers):
                    gamma += self.data[dd]['gamma_%i' % src][...,i,j] * \
                        self.data[dd][donor] * self.grid.x_to_n[donor]
            
            else:
                Gamma = self.data[dd]['Gamma'][...,i] * nabs
                
                gamma = 0.0
                for j, donor in enumerate(self.grid.absorbers):
                    gamma += self.data[dd]['gamma'][...,i,j] * \
                        self.data[dd][donor] * self.grid.x_to_n[donor]
            
            if 'Beta' in self.data[dd]:
                Beta = self.data[dd]['Beta'][...,i] * nabs * ne   
            else:
                Beta = 0 
            
            ion = Gamma + Beta + gamma # Total ionization rate
            
            # Recombinations
            alpha = self.data[dd]['alpha'][...,i] * nion * ne
            xi = self.data[dd]['xi'][...,i] * nion * ne
            recomb = alpha + xi    
                
            ax.loglog(self.grid.r_mid / cm_per_kpc, ion, 
                color=color, ls=ls, label='Total')
                
            if not total_only:      
                ax.loglog(self.grid.r_mid / cm_per_kpc, Gamma, 
                    color=color, ls='--', label=r'$\Gamma$')
                ax.loglog(self.grid.r_mid / cm_per_kpc, gamma, 
                    color=color, ls=':', label=r'$\gamma$')
                ax.loglog(self.grid.r_mid / cm_per_kpc, Beta, 
                    color=color, ls='-.', label=r'$\beta$')
                    
            if plot_recomb:
                ax.loglog(self.grid.r_mid / cm_per_kpc, recomb, 
                    color = 'b', ls = '-', label = 'Recomb.')
                    
                if not total_only:    
                    ax.loglog(self.grid.r_mid / cm_per_kpc, alpha, 
                        color = 'b', ls = '--', label = r'$\alpha$')
                    ax.loglog(self.grid.r_mid / cm_per_kpc, xi, 
                        color = 'b', ls = ':', label = r'$\xi$')
            
        ax.set_xlabel(r'$r \ (\mathrm{kpc})$') 
        ax.set_ylabel(r'Ionization Rate $(\mathrm{s}^{-1})$')
        ax.set_ylim(0.01 * 10**np.floor(np.log10(np.min(ion))), 
            10**np.ceil(np.log10(np.max(ion))))
        
        if legend:
            ax.legend(frameon=False, ncol=2, loc='best')
        
        pl.draw()    
        
        return ax
        
    def HeatingRate(self, t=1, color='r', ls='-', legend=True, src=0, 
        plot_cooling=False, label=None, ax=None):
        """
        Plot heating rate as a function of radius.
        
        Parameters
        ----------
        t : int, float
            Time (code units) at which to plot heating rate.
        plot_cooling : bool
            Plot total cooling rate vs. radius as well?
            
        """ 
        
        if type(t) is not list:
            t = [t]
        
        if ax is None:
            ax = pl.subplot(111)    
            
        if label is None:
            heat_label = r'$\mathcal{H}_{\mathrm{tot}}$'    
        else:
            heat_label = label    
        
        ax = pl.subplot(111)
        for dd in self.data.keys():
            if self.data[dd]['time'] / self.pf['time_units'] not in t: 
                continue
            
            ne = self.data[dd]['de']   
            heat, zeta, eta, psi, omega, cool = np.zeros([6, self.grid.dims])
            for absorber in self.grid.absorbers:                
                i = self.grid.absorbers.index(absorber)            
                            
                nabs = self.data[dd][absorber] * self.grid.x_to_n[absorber]
                nion = self.data[dd]['h_2'] * self.grid.x_to_n[absorber]
                            
                # Photo-heating
                if 'Heat_0' in self.data[dd].keys():
                    heat = heat + self.data[dd]['Heat_%i' % src][...,i] * nabs
                else:
                    heat = heat + self.data[dd]['Heat'][...,i] * nabs
                
                # Cooling
                zeta = zeta + self.data[dd]['zeta'][...,i] * nabs * ne # collisional ionization                
                eta = eta + self.data[dd]['eta'][...,i] * nion * ne   # recombination
                psi = psi + self.data[dd]['psi'][...,i] * nabs * ne   # collisional excitation
            
                if absorber == 'he_2':
                    omega = self.data[dd]['omega'] * nion * ne # dielectric
            
            cool = (zeta + eta + psi + omega)    
            #if self.pf['CosmologicalExpansion']:
            #    cool += self.data[dd].hubble * 3. * self.data[dd].T * k_B * self.data[dd].n_B

            mi = min(np.min(heat), np.min(cool))    
            ma = max(np.max(heat), np.max(cool))    
                
            ax.loglog(self.grid.r_mid / cm_per_kpc, heat, 
                color = color, ls = ls, label = heat_label)
            
            if plot_cooling:
                ax.loglog(self.grid.r_mid / cm_per_kpc, cool, 
                    color = 'b', ls = '-', label = r'$\mathcal{C}_{\mathrm{tot}}$')
                ax.loglog(self.grid.r_mid / cm_per_kpc, zeta, 
                    color = 'g', ls = '--', label = r'$\zeta$')
                ax.loglog(self.grid.r_mid / cm_per_kpc, psi, 
                    color = 'g', ls = ':', label = r'$\psi$')
                ax.loglog(self.grid.r_mid / cm_per_kpc, eta, 
                    color = 'c', ls = '--', label = r'$\eta$')
            
                if 'he_2' in self.grid.absorbers:
                    ax.loglog(self.grid.r_mid / cm_per_kpc, omega, 
                        color = 'c', ls = ':', label = r'$\omega_{\mathrm{HeII}}$')
                        
                #if self.pf['CosmologicalExpansion']:
                #    self.ax.loglog(self.data[dd].r / cm_per_kpc, 
                #        self.data[dd]['hubble'] * 3. * self.data[dd].T * k_B * self.data[dd].n_B, 
                #        color = 'm', ls = '--', label = r'$H(z)$')
                    
            if plot_cooling:
                ax_label = r'Heating & Cooling Rate $(\mathrm{erg/s/cm^3})$'        
            else:    
                ax_label = r'Heating Rate $(\mathrm{erg} \ \mathrm{s}^{-1} \ \mathrm{cm}^{-3})$'    
                            
        ax.set_xlabel(r'$r \ (\mathrm{kpc})$')
        ax.set_ylabel(ax_label)
        ax.set_ylim(0.001 * 10**np.floor(np.log10(mi)), 
            10**np.ceil(np.log10(ma)))
        
        if legend:
            ax.legend(frameon=False, ncol=3, loc='best')
        
        pl.draw()    
        
        self.heat = heat
        self.cool, self.zeta, self.eta, self.psi = (cool, zeta, eta, psi)
    
        return ax

    def snapshot(self, t):
        """
        Return data for snapshot at given time (in time_units).
        """
        
        if hasattr(self, 'pf'):
            t_time_units = t / self.pf['time_units']
        
        for dd in self.data.keys():
            t_time_units = self.data[dd]['time'] / self.pf['time_units']
            if t_time_units != t: 
                continue
                
            return self.data[dd]
                