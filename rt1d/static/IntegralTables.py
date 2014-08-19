"""

IntegralTables.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-10-25.

Description: Tabulate integrals that appear in the rate equations.
     
"""

import numpy as np
from ..util import ProgressBar
from ..physics.Constants import erg_per_ev
from ..physics.SecondaryElectrons import *
import os, re, scipy, itertools, math, copy
from scipy.integrate import quad, trapz, simps

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1

E_th = [13.6, 24.6, 54.4]

scipy.seterr(all='ignore')

class IntegralTable: 
    def __init__(self, pf, source, grid, logN=None):
        """
        Initialize a table of integral quantities.
        """
        self.pf = pf
        self.src = source
        self.grid = grid
        
        # need to not do this stuff if a table was supplied via source_table
        # best way: save pf in hdf5 file, use it 
        # just override table_* parameters...what else?
        
        
        #if self.pf['source_table']:
        #    self.load(self.pf['source_table'])
            
        # Move this stuff to TableProperties    
        if logN is None:
            
            # Required bounds of table assuming minimum species fraction
            self.logNlimits = self.TableBoundsAuto(self.pf['tables_xmin'])
                        
            # Only override automatic table properties if the request table size
            # is *bigger* than the default one.
            self.N = []
            self.logN = []
            for i, absorber in enumerate(self.grid.absorbers):
                
                if self.pf['tables_logNmin'][i] is not None:
                    self.logNlimits[i][0] = self.pf['tables_logNmin'][i]
                
                if self.pf['tables_logNmax'][i] is not None:
                    self.logNlimits[i][1] = self.pf['tables_logNmax'][i]
                
                logNmin, logNmax = self.logNlimits[i]
                
                d = int((logNmax - logNmin) / self.pf['tables_dlogN'][i]) + 1
            
                self.logN.append(np.linspace(logNmin, logNmax, d))
                self.N.append(np.logspace(logNmin, logNmax, d))
        else:    
            self.logN = logN
            self.N = [10**tmp for tmp in self.logN]
            
        # Retrieve dimensions, add some for secondary electrons if necessary                        
        self.dimsN = np.array([len(element) for element in self.N])
        self.elements_per_table = np.prod(self.dimsN)
        self.Nd = len(self.dimsN)
        
        self.logx = np.array([-np.inf])
        if self.pf['secondary_ionization'] > 1:
            self.esec = SecondaryElectrons(method=self.pf['secondary_ionization'])
            if self.pf['secondary_ionization'] == 2:
                self.logx = np.linspace(self.pf['tables_logxmin'], 0,
                    abs(self.pf['tables_logxmin']) \
                    / self.pf['tables_dlogx'] + 1)
                self.E = np.linspace(self.src.Emin, self.src.Emax,
                    (self.src.Emax - self.src.Emin) \
                    / self.pf['tables_dE'] + 1)
            elif self.pf['secondary_ionization'] == 3:
                self.logx = self.esec.logx
                self.E = self.esec.E
                
        self.x = 10**self.logx
            
        # Times
        if self.pf['spectrum_evolving']:
            if self.pf['tables_times'] is None:
                stop = self.pf['stop_time'] * self.pf['time_units']
                self.t = np.linspace(0, stop, 1 + stop / self.pf['tables_dt'])
            else:
                self.t = self.pf['tables_times']  
        else:
            self.t = np.array([0])
                                                
        # What quantities are we going to compute?
        self.IntegralList = self.ToCompute()
        
        # Create array of all combinations of column densities and
        # corresponding indices
        self.TableProperties()
        
        self.E_th = {}
        for absorber in ['h_1', 'he_1', 'he_2']:
            self.E_th[absorber] = self.grid.ioniz_thresholds[absorber]
        
    def TableBoundsAuto(self, xmin=1e-5):
        """
        Calculate what the bounds of the table must be for a given grid.
        """
        
        logNlimits = []
        for i, absorber in enumerate(self.grid.absorbers):
            n = self.grid.species_abundances[absorber] * self.grid.n_H
            logNmin = math.floor(np.log10(xmin[i] * np.min(n) * np.min(self.grid.dr)))
            logNmax = math.ceil(np.log10(np.sum(n * self.grid.dr)))
            logNlimits.append([logNmin, logNmax])
            
        return logNlimits
        
    def ToCompute(self):
        """
        Return list of quantities to compute.
        """    

        integrals = ['Tau', 'Phi']
        if not self.grid.isothermal:
            integrals.append('Psi')
        
        if self.pf['secondary_ionization'] >= 2:
            integrals.extend(['PhiWiggle', 'PhiHat'])
            
            if not self.grid.isothermal:
                integrals.extend(['PsiWiggle', 'PsiHat'])
                integrals.remove('Psi')
        
        return integrals
        
    def TableProperties(self):
        """
        Figure out ND space of all lookup table elements.
        """  
          
        # Determine indices for column densities.       
        tmp = []
        for dims in self.dimsN:
            tmp.append(np.arange(dims))
                
        if rank == 0:
            print "Setting up integral table..."
                    
        # Values that correspond to indices
        logNarr = []
        for item in itertools.product(*self.logN):
            logNarr.append(item)
        
        # Indices for column densities
        iN = []
        for item in itertools.product(*tmp):
            iN.append(tuple(item))    
            
        #iN = np.indices(self.dimsN)
            
        self.indices_N = iN
        self.logNall = np.array(logNarr)
        self.Nall = 10**self.logNall
        
        self.axes = copy.copy(self.logN)
        self.axes_names = []
        for absorber in self.grid.absorbers:
            self.axes_names.append('logN_%s' % absorber)
        
        # Determine indices for ionized fraction and time.
        if self.pf['secondary_ionization'] > 1:
            self.Nd += 1
            self.axes.append(self.logx)
            self.axes_names.append('x')
        if self.pf['spectrum_evolving']:
            self.Nd += 1
            self.axes.append(self.t)
            self.axes_names.append('t')
                                
    def DatasetName(self, integral, absorber, donor):
        """
        Return name of table.  Will be called from dictionary or hdf5 using
        this name.
        """    
        
        if integral in ['PhiWiggle', 'PsiWiggle']:
            return "log%s_%s_%s" % (integral, absorber, donor)
        elif integral == 'Tau':
            return 'log%s' % integral    
        else:
            return "log%s_%s" % (integral, absorber)   
              
    def TabulateRateIntegrals(self):
        """
        Return a dictionary of lookup tables, and also store a copy as self.itabs.
        """
        
        if rank == 0:
            print '\nTabulating integral quantities...'        
                
        # Loop over integrals
        h = 0
        tabs = {}
        i_donor = 0
        while h < len(self.IntegralList):
            integral = self.IntegralList[h]
                        
            donor = self.grid.absorbers[i_donor]
            for i, absorber in enumerate(self.grid.absorbers):
                
                name = self.DatasetName(integral, absorber, donor)
                                        
                if integral == 'Tau' and i > 0:
                    continue
                    
                # Don't know what to do with metal photo-electron energy    
                if re.search('Wiggle', name) and absorber in self.grid.metals:
                    continue
                    
                dims = list(self.dimsN.copy())
                
                if integral == 'Tau':
                    dims.append(1)
                else:    
                    dims.append(len(self.t))
                if self.pf['secondary_ionization'] > 1 \
                    and integral not in ['Tau', 'Phi']:
                    dims.append(len(self.logx))
                else:
                    dims.append(1)
                
                pb = ProgressBar(self.elements_per_table, name)                
                pb.start()
                                                              
                tab = np.zeros(dims)
                for j, ind in enumerate(self.indices_N):
                        
                    if j % size != rank:
                        continue
                        
                    tmpt = self.t
                    tmpx = self.x                                    
                    if integral == 'Tau':
                        tmpx = [0]   
                        tmpt = [0]
                    if integral == 'Phi':
                        tmpx = [0]      
                        
                    for k, t in enumerate(tmpt):                        
                        for l, x in enumerate(tmpx):  
                            tab[ind][k,l] = self.Tabulate(integral, 
                                absorber, donor, self.Nall[j], x=x, t=t)

                    pb.update(j)
                    
                tabs[name] = np.squeeze(tab).copy()
                
                pb.finish()
                
            if re.search('Wiggle', name):
                if self.grid.metals:
                    if self.grid.absorbers[i_donor + 1] in self.grid.metal_ions:
                        h += 1
                    else:
                        i_donor += 1
                elif (i_donor + 1) == len(self.grid.absorbers):    
                    i_donor = 0
                    h += 1
                else:
                    i_donor += 1
            else:
                h += 1
                i_donor = 0
                       
        if rank == 0:                        
            print 'Integral tabulation complete.'
            
        # Collect results from all processors    
        if size > 1:        
            collected_tabs = {}
            for tab in tabs:
                tmp = np.zeros_like(tabs[tab])
                nothing = MPI.COMM_WORLD.Allreduce(tabs[tab], tmp)
                collected_tabs[tab] = tmp.copy()
                del tmp
                
            tabs = collected_tabs.copy()
        
        self.tabs = tabs
        return tabs         
            
    def TotalOpticalDepth(self, N):
        """
        Optical depth due to all absorbing species at given column density.
        Assumes ncol is a 3-element array.
        """
        
        tau = 0.0
        for absorber in self.grid.absorbers:
            tau += self.OpticalDepth(N[self.grid.absorbers.index(absorber)], 
                absorber)
                
            #if self.grid.approx_helium:
            #    Nhe = self.grid.abundances[1] \
            #        * N[self.grid.absorbers.index('h_1')]
            #    tau += self.OpticalDepth(Nhe, 'he_1')  
    
        return tau
               
    def OpticalDepth(self, N, absorber):
        """
        Optical depth of species integrated over entire spectrum at a 
        given column density.  We just use this to determine which cells
        are inside/outside of an I-front (OpticalDepthDefiningIfront = 0.5
        by default).
        """        
                                
        if self.src.continuous:
            integrand = lambda E: self.PartialOpticalDepth(E, N, absorber)                           
            result = quad(integrand, 
                max(self.E_th[absorber], self.src.Emin),
                self.src.Emax)[0]
        else:                                                                                                                                                                                
            result = np.sum(self.PartialOpticalDepth(self.src.E, N, species)[self.src.E > E_th[species]])
            
        return result
        
    def PartialOpticalDepth(self, E, N, absorber):
        """
        Returns the optical depth at energy E due to column density ncol of species.
        """
                        
        return self.grid.bf_cross_sections[absorber](E) * N
        
    def SpecificOpticalDepth(self, E, N):
        """
        Returns the optical depth at energy E due to column densities 
        of all absorbers.
        """
                                    
        if type(E) in [float, np.float32, np.float64]:
            E = [E]
                                           
        tau = np.zeros_like(E)
        for j, energy in enumerate(E):
            tmp = 0
            for i, absorber in enumerate(self.grid.absorbers):
                if energy >= self.grid.ioniz_thresholds[absorber]:
                    if type(N) is dict:
                        tmp += self.PartialOpticalDepth(energy, N[absorber], 
                            absorber)
                    else:
                        tmp += self.PartialOpticalDepth(energy, N[i], 
                            absorber)
            tau[j] = tmp
            del tmp
        
        return tau
        
    def Tabulate(self, integral, absorber, donor, N, x=None, t=0):
        if integral == 'Phi':
            table = self.Phi(N, absorber, t = t)
        if integral == 'Psi':
            table = self.Psi(N, absorber, t = t)
        if integral == 'PhiWiggle':
            table = self.PhiWiggle(N, absorber, donor, x = x, t = t)
        if integral == 'PsiWiggle':
            table = self.PsiWiggle(N, absorber, donor, x = x, t = t)
        if integral == 'PhiHat':
            table = self.PhiHat(N, absorber, donor, x = x, t = t)
        if integral == 'PsiHat':
            table = self.PsiHat(N, absorber, donor, x = x, t = t)
        if integral == 'Tau':
            table = self.TotalOpticalDepth(N)
            
        return np.log10(table)
        
    def Phi(self, N, absorber, t=0):
        """
        Equation 10 in Mirocha et al. 2012.
        """
                                                         
        # Otherwise, continuous spectrum                
        if self.pf['photon_conserving']:
            integrand = lambda E: self.src.Spectrum(E, t=t) * \
                np.exp(-self.SpecificOpticalDepth(E, N)[0]) / E
                            
        else:
            integrand = lambda E: self.grid.bf_cross_sections[absorber](E) * \
                self.src.Spectrum(E, t = t) * \
                np.exp(-self.SpecificOpticalDepth(E, N)[0]) / E \
                / self.E_th[absorber]
                
        integral = quad(integrand, max(self.E_th[absorber], self.src.Emin), 
            self.src.Emax, limit=1000)[0] / erg_per_ev    
            
        if not self.pf['photon_conserving']:
            integral *= self.E_th[absorber]
            
        return integral 
        
    def Psi(self, N, absorber, t=None):            
        """
        Equation 11 in Mirocha et al. 2012.
        """        
        
        # Otherwise, continuous spectrum    
        if self.pf['photon_conserving']:
            integrand = lambda E: self.src.Spectrum(E, t = t) * \
                np.exp(-self.SpecificOpticalDepth(E, N)[0])
        else:
            integrand = lambda E: self.grid.bf_cross_sections[absorber](E) * \
                self.src.Spectrum(E, t = t) * \
                np.exp(-self.SpecificOpticalDepth(E, N)[0]) \
                / self.E_th[absorber]
        
        integral = quad(integrand, max(self.E_th[absorber], self.src.Emin), 
            self.src.Emax, limit=1000)[0]
            
        if not self.pf['photon_conserving']:
            integral *= self.E_th[absorber]
        
        return integral
                              
    def PhiHat(self, N, absorber, donor=None, x=None, t=None):
        """
        Equation 2.20 in the manual.
        """        
        
        Ei = self.E_th[absorber]
        
        # Otherwise, continuous spectrum                
        if self.pf['photon_conserving']:
            integrand = lambda E: \
                self.esec.DepositionFraction(x,E=E-Ei, channel='heat') * \
                self.src.Spectrum(E, t = t) * \
                np.exp(-self.SpecificOpticalDepth(E, N)[0]) / E
        else:
            integrand = lambda E: \
                self.esec.DepositionFraction(x, E=E-Ei, channel='heat') * \
                PhotoIonizationCrossSection(E, absorber) * \
                self.src.Spectrum(E, t = t) * \
                np.exp(-self.SpecificOpticalDepth(E, N)[0]) / E \
                / self.E_th[absorber]    
        
        # Integrate over energies in lookup table
        c = self.E >= max(Ei, self.src.Emin)
        c &= self.E <= self.src.Emax                       
        samples = np.array([integrand(E) for E in self.E[c]])[..., 0]
             
        integral = simps(samples, self.E[c]) / erg_per_ev     
        
        if not self.pf['photon_conserving']:
            integral *= self.E_th[absorber]
            
        return integral
                
    def PsiHat(self, N, absorber, donor=None, x=None, t=None):            
        """
        Equation 2.21 in the manual.
        """        
        
        Ei = self.E_th[absorber]
        
        # Otherwise, continuous spectrum    
        if self.pf['photon_conserving']:
            integrand = lambda E: \
                self.esec.DepositionFraction(x, E=E-Ei, channel='heat') * \
                self.src.Spectrum(E, t = t) * \
                np.exp(-self.SpecificOpticalDepth(E, N)[0])
        else:
            integrand = lambda E: \
                self.esec.DepositionFraction(x, E=E-Ei, channel='heat') * \
                PhotoIonizationCrossSection(E, species) * \
                self.src.Spectrum(E, t = t) * \
                np.exp(-self.SpecificOpticalDepth(E, N)[0]) \
                / self.E_th[absorber]
        
        # Integrate over energies in lookup table
        c = self.E >= max(Ei, self.src.Emin)
        c &= self.E <= self.src.Emax
        samples = np.array([integrand(E) for E in self.E[c]])[..., 0]
        
        integral = simps(samples, self.E[c])  
        
        if not self.pf['photon_conserving']:
            integral *= self.E_th[absorber]
        
        return integral
            
    def PhiWiggle(self, N, absorber, donor, x=None, t=None):
        """
        Equation 2.18 in the manual.
        """        
        
        Ej = self.E_th[donor]
        
        # Otherwise, continuous spectrum                
        if self.pf['photon_conserving']:
            integrand = lambda E: \
                self.esec.DepositionFraction(x, E=E-Ej, channel=absorber) * \
                self.src.Spectrum(E, t = t) * \
                np.exp(-self.SpecificOpticalDepth(E, N)[0]) / E

        #else:
        #    integrand = lambda E: 1e10 * \
        #        self.esec.DepositionFraction(E, xHII, channel = species + 1) * \
        #        PhotoIonizationCrossSection(E, species) * \
        #        self.src.Spectrum(E, t = t) * \
        #        np.exp(-self.SpecificOpticalDepth(E, ncol)[0]) / E \
        #        / self.E_th[absorber]
        
        # Integrate over energies in lookup table    
        c = self.E >= max(Ej, self.src.Emin)
        c &= self.E <= self.src.Emax
        samples = np.array([integrand(E) for E in self.E[c]])[..., 0]
        
        integral = simps(samples, self.E[c]) / erg_per_ev
            
        if not self.pf['photon_conserving']:
            integral *= self.E_th[absorber]
                                        
        return integral
                              
    def PsiWiggle(self, N, absorber, donor, x=None, t=None):            
        """
        Equation 2.19 in the manual.
        """        
        
        Ej = self.E_th[donor]
        
        # Otherwise, continuous spectrum    
        if self.pf['photon_conserving']:
            integrand = lambda E: \
                self.esec.DepositionFraction(x, E=E-Ej, channel=absorber) * \
                self.src.Spectrum(E, t = t) * \
                np.exp(-self.SpecificOpticalDepth(E, N)[0])
        #else:
        #    integrand = lambda E: PhotoIonizationCrossSection(E, species) * \
        #        self.src.Spectrum(E, t = t) * \
        #        np.exp(-self.SpecificOpticalDepth(E, ncol)[0])
        #        / self.E_th[absorber]
           
        # Integrate over energies in lookup table        
        c = self.E >= max(Ej, self.src.Emin)
        c &= self.E <= self.src.Emax
        samples = np.array([integrand(E) for E in self.E[c]])[..., 0]
             
        integral = simps(samples, self.E[c])
          
        if not self.pf['photon_conserving']:
            integral *= self.E_th[absorber]    
                
        return integral
            
    def PsiBreve(self, N, absorber, donor, x=None, t=None):
        """
        Return fractional Lyman-alpha excitation.
        """         
        
        pass
                                  
    def save(self, fn=None):
        """ Write table to hdf5. """
        
        if rank > 0:
            return
        
        try:
            import h5py
            have_h5py = True
        except ImportError:
            have_h5py = False    
            
        have_h5py = False # testing
            
        if fn is None and have_h5py:
            fn = 'rt1d_integral_table.hdf5'
        elif fn is None:
            fn = 'rt1d_integral_table.npz'
        
        if have_h5py:
        
            f = h5py.File(fn)
            for i, axis in enumerate(self.axes):
                ds = f.create_dataset(self.axes_names[i], data=axis)
                ds.attrs.create('axis', data=i)
                ds.attrs.create('logN', data=int(self.axes_names[i][0:4]=='logN'))
            
            for tab in self.tabs:
                f.create_dataset(tab, data=self.tabs[tab])
                
            # Save parameter file
            pf_grp = f.create_group('parameters')
            for par in self.pf:
                if self.pf[par] is not None:
                    try:
                        pf_grp.create_dataset(par, data=self.pf[par])
                    except TypeError:
                        if type(self.pf[par]) is list:
                            Nones = 0
                            for i in range(len(self.pf[par])):
                                if self.pf[par] is None:
                                    Nones += 1
                            
                            if Nones == len(self.pf[par]):
                                pf_grp.create_dataset(par, data=[-99999] * Nones)
                        else:
                            raise ValueError('Dunno what to do here.')        
                        
                        
                else:
                    pf_grp.create_dataset(par, data=-99999)
                
            f.close()
            
        else:
            f = open(fn, 'w')

            kw = {tab:self.tabs[tab] for tab in self.tabs}
            for i, axis in enumerate(self.axes):
                kw.update({self.axes_names[i]: self.axes[i]})
                
            np.savez(f, **kw)
                
            f.close()  
            
            # Save parameters
            import pickle
            
            prefix = fn[0:fn.rfind('.')]
            suffix = fn[fn.rfind('.')+1:]
            
            f = open('%s.pars.pkl' % prefix, 'wb')
            pickle.dump(self.pf, f)
            f.close()
            if rank == 0:
                print 'Wrote %s and %s.pars.pkl' % (fn, prefix)  

    def load(self, fn):
        """
        Load table from hdf5. 
        """
        
        if re.search('npz', fn):
                        
            data = np.load(fn)
        
            axes = []
            self.tabs = {}
            for key in data.keys():
                if re.search('logN', key):
                    if re.search('h_1', key):
                        i = 0
                    else:
                        raise NotImplemented('help! fix npz reader for N > 1')
                    
                    axes.append([i, key, data[key]])
        
                    continue
            
                self.tabs[key] = data[key]

        else:
            axes = []
            self.tabs = {}
            f = h5py.File(fn)
            for element in f.keys():
                if f[element].attrs.get('axis') is not None:
                    axes.append([int(f[element].attrs.get('axis')), element, 
                        f[element].value])
                    continue
                
                self.tabs[element] = f[element].value
            
            f.close()
        
        print 'Read integral table from %s.' % fn
        
        axis_nums, axis_names, values = zip(*axes)

        # See if parameter file and integral table are consistent
        ok = True
        for i, axis in enumerate(axis_names):

            if axis not in self.axes_names:
                print "WARNING: Axis \'%s\' not expected." % axis
                continue

            if np.all(np.array(values[i]) == self.axes[i]):
                continue
            
            print 'WARNING: Axis \'%s\' has %i elements. Expected %i.' \
                % (axis, np.array(values[i]).size, self.axes[i].size)
            ok = False
            
        if not ok:
            raise ValueError('Axes of integral table inconsistent!')    
                
        return self.tabs
               
            