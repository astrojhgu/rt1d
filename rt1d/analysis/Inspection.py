"""

Inspection.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Mar 29 10:45:23 2012

Description: This is called 'Inspection' because I mostly use it to make sure
stuff is correct (like integral tables, interpolation, solver speed, etc.), 
rather than analyzing data to make science-motivated figures.

"""

import os
import numpy as np
import pylab as pl

class Inspect:
    def __init__(self, anl):
        self.pf = anl.pf
        self.anl = anl
        self.data = self.anl.data
        self.itabs = self.anl.itabs
        self.interp = self.anl.interp
                
    def InspectIntegralTable(self, intnum = 1, species = 0, donor_species = 0,
        nHI = None, nHeI = 0.0, nHeII = 0.0,
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
        if intnum == 1:
            integral = 'Phi%i' % species 
            ylabel = r'$\Phi_{\mathrm{%s}}$' % s
        elif intnum == 1.1:
            integral = 'PhiWiggle%i%i' % (species, donor_species)
            ylabel = r'$\widetilde{\Phi}_{\mathrm{%s}}$' % s 
        elif intnum == 1.2:
            integral = 'PhiHat%i' % species
            ylabel = r'$\widetilde{\Phi}_{\mathrm{%s}}$' % s        
        elif intnum == 2:
            integral = 'Psi%i' % species
            ylabel = r'$\Psi_{\mathrm{%s}}$' % s
        elif intnum == 2.1:
            integral = 'PsiWiggle%i%i' % (species, donor_species)
            ylabel = r'$\widetilde{\Psi}_{\mathrm{%s}}$' % s 
        elif intnum == 2.2:
            integral = 'PsiHat%i' % species
            ylabel = r'$\widetilde{\Psi}_{\mathrm{%s}}$' % s            
        else:
            integral = 'TotalOpticalDepth' 
            ylabel = r'$\sum_i\int_{\nu} \tau_{i,\nu} d\nu$'
            pl.rcParams['figure.subplot.left'] = 0.2012
        
        # Figure out axes
        if nHI is None:
            x = self.itabs['HIColumnValues_x']
            if self.pf.MultiSpecies:
                i1 = np.argmin(np.abs(self.itabs['HeIColumnValues_y'] - nHeI))
                i2 = np.argmin(np.abs(self.itabs['HeIIColumnValues_z'] - nHeII))
                y = self.itabs[integral][0:,i1,i2]
            else:
                if intnum % int(intnum) != 0:
                    y = self.itabs[integral][:,0]
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
        
    def InspectInterpolation(self, intnum = 1, species = 0, donor_species = 0,
        nHI = np.linspace(12, 19, 1000), 
        nHeI = 0.0, nHeII = 0.0, color = 'b', ls = '-', x_HII = 1e-4):
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
        if intnum == 1:
            integral = 'Phi%i' % species 
            ylabel = r'$\Phi_{\mathrm{%s}}$' % s
        elif intnum == 1.1:
            integral = 'PhiWiggle%i%i' % (species, donor_species)
            ylabel = r'$\widetilde{\Phi}_{\mathrm{%s}}$' % s 
        elif intnum == 1.2:
            integral = 'PhiHat%i' % species
            ylabel = r'$\widetilde{\Phi}_{\mathrm{%s}}$' % s        
        elif intnum == 2:
            integral = 'Psi%i' % species
            ylabel = r'$\Psi_{\mathrm{%s}}$' % s
        elif intnum == 2.1:
            integral = 'PsiWiggle%i%i' % (species, donor_species)
            ylabel = r'$\widetilde{\Psi}_{\mathrm{%s}}$' % s 
        elif intnum == 2.2:
            integral = 'PsiHat%i' % species
            ylabel = r'$\widetilde{\Psi}_{\mathrm{%s}}$' % s            
        else:
            integral = 'TotalOpticalDepth' 
            ylabel = r'$\sum_i\int_{\nu} \tau_{i,\nu} d\nu$'
            pl.rcParams['figure.subplot.left'] = 0.2012
        
        result = []
        if type(nHI) not in [int, float]:
            x = nHI
            for col in nHI:
                tmp = [col, nHeI, nHeII]                
                indices = self.interp.GetIndices(tmp, x_HII = x_HII)                
                result.append(self.interp.interp(indices, 
                    '%s' % integral, tmp, x_HII = x_HII))
            
        elif type(nHeI) not in [int, float]:
            x = nHeI
            for col in nHeI:
                tmp = [nHI, col, nHeII]
                result.append(self.interp.interp(self.interp.GetIndices(tmp), 
                    '%s' % integral))    
        
        else:
            x = nHeII
            for col in nHeII:
                tmp = [nHI, nHeI, col]
                result.append(self.interp.interp(self.interp.GetIndices(tmp), 
                    '%s' % integral))            
            
        self.ax = pl.subplot(111)
        self.ax.loglog(10**np.array(x), result, color = color, ls = ls)
                                
        pl.draw()                    
            
    def InspectTimestepEvolution(self, t = 1, legend = True):
        """
        Plot timestep as a function of radius.
        """    
        
        if legend and hasattr(self, 'ax'):
            legend = False
        
        self.dt = np.zeros([5, self.GridDimensions])
        for dd in self.data.keys():
            if self.data[dd].t / self.pf.TimeUnits != t: 
                continue
        
            dHIIdt = self.data[dd].n_HI * \
                (self.data[dd].Gamma[0] + self.data[dd].Beta[0] * self.data[dd].n_e) \
                + np.sum(self.data[dd].gamma[0] * self.data[dd].nabs) \
                - self.data[dd].n_HII * self.data[dd].n_e * self.data[dd].alpha[0]
                
            dHeIdt = self.data[dd].n_HeII * \
                (self.data[dd].alpha[1] + self.data[dd].xi[1]) * self.data[dd].n_e \
                - self.data[dd].n_HeI * (self.data[dd].Gamma[1] + self.data[dd].Beta[1]) \
                - np.sum(self.data[dd].gamma[1] * self.data[dd].nabs) \
                * self.data[dd].n_e     
                
            dHeIIdt = self.data[dd].n_HeI * \
                (self.data[dd].Gamma[1] + self.data[dd].Beta[1] * self.data[dd].n_e) \
                + np.sum(self.data[dd].gamma[1] * self.data[dd].nabs) \
                + self.data[dd].alpha[2] * self.data[dd].n_HeIII * self.data[dd].n_e \
                - self.data[dd].n_HeII * self.data[dd].n_e * \
                (self.data[dd].alpha[1] + self.data[dd].Beta[2] + self.data[dd].xi[1]) 
                
            dHeIIIdt = self.data[dd].n_HeII * \
                (self.data[dd].Gamma[2] + self.data[dd].Beta[2] * self.data[dd].n_e) \
                - self.data[dd].n_HeIII * self.data[dd].n_e * self.data[dd].alpha[2]
                
            defdt = dHIIdt + dHeIIdt + 2. * dHeIIIdt
                                        
            self.dt[0] = self.pf.MaxHIIChange * self.data[dd].n_HI / np.abs(dHIIdt) / self.pf.TimeUnits
            self.dt[1] = self.pf.MaxHeIChange * self.data[dd].n_HeI / np.abs(dHeIdt) / self.pf.TimeUnits
            self.dt[2] = self.pf.MaxHeIIChange * self.data[dd].n_HeII / np.abs(dHeIIdt) / self.pf.TimeUnits
            self.dt[3] = self.pf.MaxHeIIIChange * self.data[dd].n_HeIII / np.abs(dHeIIIdt) / self.pf.TimeUnits
            self.dt[4] = self.pf.MaxElectronChange * (self.data[dd].n_e / self.data[dd].n_B) / np.abs(defdt) / self.pf.TimeUnits  
            
            break
            
        mi = 0.01 * 10**np.floor(np.log10(np.min(self.dt)))
        ma = 1e2
                        
        self.ax = pl.subplot(111)
        self.ax.loglog(self.data[dd].r / self.pf.LengthUnits, self.dt[0], color = 'k', label = r'$\Delta t_{\mathrm{HII}}$')
        i1 = np.argmin(np.abs(self.data[dd].tau[0] - self.pf.OpticalDepthDefiningIFront[0]))
        self.ax.loglog([self.data[dd].r[i1] / self.pf.LengthUnits] * 2, [mi, ma], color = 'b', ls = '-')
        
        if self.pf.MultiSpecies:
            
            if self.pf.HeIRestrictedTimestep:
                self.ax.loglog(self.data[dd].r / self.pf.LengthUnits, self.dt[1], 
                    color = 'k', ls = '-.', label = r'$\Delta t_{\mathrm{HeI}}$')
            
            if self.pf.HeIIRestrictedTimestep:
                self.ax.loglog(self.data[dd].r / self.pf.LengthUnits, self.dt[2], 
                    color = 'k', ls = '--', label = r'$\Delta t_{\mathrm{HeII}}$')
            
            if self.pf.HeIIIRestrictedTimestep:
                self.ax.loglog(self.data[dd].r / self.pf.LengthUnits, self.dt[3], 
                    color = 'k', ls = ':', label = r'$\Delta t_{\mathrm{HeIII}}$')
            
            if self.pf.ElectronFractionRestrictedTimestep:
                self.ax.loglog(self.data[dd].r / self.pf.LengthUnits, self.dt[4], 
                    color = 'k', ls = '-.', label = r'$\Delta t_{e^-}$')
            
            i2 = np.argmin(np.abs(self.data[dd].tau[1] - self.pf.OpticalDepthDefiningIFront[1]))
            i3 = np.argmin(np.abs(self.data[dd].tau[2] - self.pf.OpticalDepthDefiningIFront[2]))
            self.ax.loglog([self.data[dd].r[i2] / self.pf.LengthUnits] * 2, [mi, ma], color = 'b', ls = '--')
            self.ax.loglog([self.data[dd].r[i3] / self.pf.LengthUnits] * 2, [mi, ma], color = 'b', ls = ':')
         
        # Minimum dtphot    
        self.ax.loglog(self.data[dd].r / self.pf.LengthUnits, self.data[dd].dtPhoton, color = 'g', ls = '-', label = r'$\Delta t_{\mathrm{min}}$')
        
        # Timestep chosen for next step   
        self.ax.loglog([self.data[dd].r[0] / self.pf.LengthUnits, self.data[dd].r[-1] / self.pf.LengthUnits], 
            [np.min(self.data[dd].dtPhoton)] * 2, color = 'r', ls = '-')        
        
        self.ax.set_xlabel(r'$r / L_{\mathrm{box}}$') 
        self.ax.set_ylabel(r'$\Delta t_{\mathrm{phot}} / \mathrm{TimeUnits}$') 
        self.ax.annotate(r'$\Delta t_{\mathrm{next}}$', 
            (self.data[dd].r[self.data[dd].dtPhoton == np.min(self.data[dd].dtPhoton)] / self.pf.LengthUnits,
             0.85 * np.min(self.data[dd].dtPhoton)),
            va = 'top', ha = 'center')
            
        self.ax.set_ylim(mi, ma) 
        
        if legend:
            self.ax.legend(loc = 'lower right', ncol = 3, frameon = False)   
        
        pl.draw()    
        
    def InspectSolverSpeed(self, t = 1, color = 'k', ls = '-', legend = True, 
        ode = True, equation = -1, norm = True):
        """
        Plot ODE step as a function of radius.
        
        Options: odeit, odeitrate, rootit.
        """    
        
        if legend and hasattr(self, 'ax'):
            legend = False
        
        for dd in self.data.keys():
            if self.data[dd].t / self.pf.TimeUnits != t: 
                continue
        
            self.ax = pl.subplot(111)
            
            if ode:
                self.ax.loglog(self.data[dd].r / self.pf.LengthUnits, self.data[dd].odeit, color = color, ls = ls, label = r'$N_{\mathrm{ODE}}$')
            
            if equation >= 0:
                if norm:
                    self.ax.loglog(self.data[dd].r / self.pf.LengthUnits, 
                        self.data[dd].rootit[0:,equation] / self.data[dd].odeit, 
                        color = color, ls = ls, label = r'$N_{\mathrm{NR}}/N_{\mathrm{ODE}}$')
                else:
                    self.ax.loglog(self.data[dd].r / self.pf.LengthUnits, 
                        self.data[dd].rootit[0:,equation], 
                        color = color, ls = ls, label = r'$N_{\mathrm{NR}}$')        
            
            self.ax.set_xlabel(r'$r / L_{\mathrm{box}}$') 
            self.ax.set_ylabel(r'$N_{\mathrm{iter}}$') 
            
            break
            
        if legend:
            self.ax.legend(loc = 'upper right', ncol = 2, frameon = False)       
            
        pl.draw()        