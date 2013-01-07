"""
InitializeIntegralTables.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-10-25.

Description: Tabulate integrals that appear in the rate equations.
     
"""

import numpy as np
from ..run import ProgressBar
import os, re, scipy, itertools, math
from scipy.integrate import quad, romberg
from ..physics.Constants import erg_per_ev
from ..physics.SecondaryElectrons import *
from scipy.interpolate import LinearNDInterpolator

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    print "Module mpi4py not found.  No worries, we'll just run in serial."
    rank = 0
    size = 1

E_th = [13.6, 24.6, 54.4]

tiny_number = 1e-30
negligible_column = 1

scipy.seterr(all = 'ignore')

class IntegralTable: 
    def __init__(self, pf, source, grid):
        self.pf = pf
        
        # Make these optional
        self.src = source
        self.grid = grid
        
        # Required bounds of table assuming minimum species fraction
        self.logNlimits = \
            self.TableBoundsAuto(self.src.SpectrumPars['smallest_x'])
        
        # Only override automatic table properties if the request table size
        # is *bigger* than the default one.
        self.N = []
        self.logN = []
        for i, absorber in enumerate(self.grid.absorbers):
            
            if self.src.SpectrumPars['logNmin'] < self.logNlimits[i][0]:
                self.logNlimits[i][0] = self.src.SpectrumPars['logNmin'][i]
            
            if self.src.SpectrumPars['logNmax'] < self.logNlimits[i][1]:
                self.logNlimits[i][1] = self.src.SpectrumPars['logNmax'][i]
            
            logNmin, logNmax = self.logNlimits[i]
            
            d = int((logNmax - logNmin) / self.src.SpectrumPars['dlogN'][i]) + 1
                        
            self.logN.append(np.linspace(logNmin, logNmax, d))
            self.N.append(np.logspace(logNmin, logNmax, d))
                                
        self.Nd = len(self.grid.Z) 
        self.dims = [len(element) for element in self.N]
        
        if self.pf['secondary_ionization'] > 1:
            self.Nd += 1
            if self.pf['secondary_ionization'] == 2:
                self.dims.append(len(self.src.SpectrumPars['dlogx']))
            elif self.pf['secondary_ionization'] == 3:
                self.esec = SecondaryElectrons(method = 3)
                self.dims.append(len(self.esec.x))
        
        # What quantities are we going to compute?
        self.IntegralList = self.ToCompute()
        
        # Create array of all combinations of column densities and
        # corresponding indices
        self.TableProperties()
                
        #self.cosm = Cosmology(pf)
        #self.esec = SecondaryElectrons(pf)
        #self.grid = InitializeGrid(pf)
                        
        # Column densities - determine automatically
        #if pf['CosmologicalExpansion']:
        #    self.HIColumnMin = np.floor(np.log10(pf['MinimumSpeciesFraction'] * self.cosm.nH0 * (1. + self.cosm.zf)**3 * min(self.grid.dx)))
        #    self.HIColumnMax = np.ceil(np.log10(self.cosm.nH0 * (1. + self.cosm.zi)**3 * pf['LengthUnits']))
        #    self.HeIColumnMin = self.HeIIColumnMin = np.floor(np.log10(10**self.HIColumnMin * self.cosm.y))
        #    self.HeIColumnMax = self.HeIIColumnMax = np.ceil(np.log10(10**self.HIColumnMax * self.cosm.y))
        #else:    
        #    self.n_H = (1. - self.cosm.Y) * self.grid.density / m_H
        #    self.n_He = self.cosm.Y * self.grid.density / m_He
        #    self.HIColumnMin = np.floor(np.log10(pf['MinimumSpeciesFraction'] * np.min(self.n_H * self.grid.dx)))
        #    self.HIColumnMax = np.ceil(np.log10(pf["LengthUnits"] * np.max(self.n_H)))
        #    self.HeIColumnMin = self.HeIIColumnMin = np.floor(np.log10(pf['MinimumSpeciesFraction'] * np.min(self.n_He * self.grid.dx)))
        #    self.HeIColumnMax = self.HeIIColumnMax = np.ceil(np.log10(pf['LengthUnits'] * np.max(self.n_He)))            
        
        # Override limits if allowing optically thin approx
        #if pf['AllowSmallTauApprox']:
        #    self.HIColumnMin = pf['OpticallyThinColumn'][0]
        #    self.HeIColumnMin = self.HeIIColumnMin = pf['OpticallyThinColumn'][1]
            
            
            
        #self.HINBins = pf['ColumnDensityBinsHI']
        #self.HeINBins = pf['ColumnDensityBinsHeI']
        #self.HeIINBins = pf['ColumnDensityBinsHeII']
        #                
        #self.HIColumn = np.linspace(self.HIColumnMin, self.HIColumnMax, self.HINBins)
        #
        #self.itabs = None

        # Set up column density vectors for each absorber
        #if self.pf['MultiSpecies'] > 0: 
        #    self.HeIColumn = np.linspace(self.HeIColumnMin, self.HeIColumnMax, self.HeINBins)
        #    self.HeIIColumn = np.linspace(self.HeIIColumnMin, self.HeIIColumnMax, self.HeIINBins)        
        #else:
        #    self.HeIColumn = np.ones_like(self.HIColumn) * tiny_number
        #    self.HeIIColumn = np.ones_like(self.HIColumn) * tiny_number
        
        
        
        # What will our table look like?
        #self.Nd, self.Nt, self.dims, self.values, \
        #self.indices, self.columns, self.dcolumns, self.locs, \
        #self.colnames = \
        #    self.TableProperties()
                        
        # Retrive rt1d environment - look for tables in rt1d/input
        self.rt1d = os.environ.get("RT1D")

        #if pf['DiscreteSpectrum']:
        #    self.zeros = np.zeros_like(self.src.E)
        #else:
        #    self.zeros = np.zeros(1)
            
        #self.tname = self.DetermineTableName() 
        
    def TableBoundsAuto(self, xmin = 1e-5):
        """
        Calculate what the bounds of the table must be for a 
        given grid.
        """
        
        logNlimits = []
        for i, absorber in enumerate(self.grid.absorbers):
            n = self.grid.species_abundances[absorber] * self.grid.n_H
            logNmin = math.floor(np.log10(xmin[i] * np.min(n) * np.min(self.grid.dr)))
            logNmax = math.ceil(np.log10(np.sum(n * self.grid.dr)))
            logNlimits.append((logNmin, logNmax))
            
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
        
        Ns = len(self.grid.absorbers)

        Nt = 1. * Ns            # Number of tables (unique quantities)
        
        if not self.grid.isothermal:
            Nt += 1 * Ns
        
        Nd = len(self.grid.absorbers)
                                    
        if self.pf['secondary_ionization'] >= 2:
            Nd += 1
            Nt += 2. * (1. + 8 * self.pf['MultiSpecies'])  # Phi/Psi Wiggle
            Nt += 2. * Ns                                  # Phi/Psi Hat
            
        #if self.pf['SourceTimeEvolution']:
        #    Nd += 1
        #    dims.append(self.pf['AgeBins'])
        #    columns.append(self.src.Age)
        #    colnames.append('Age')
        #    locs.append(4)
            
        indices = []
        for dims in self.dims:
            indices.append(np.arange(dims))
                
        tmp1 = itertools.product(*self.logN)
        tmp2 = itertools.product(*indices)
        
        values = []
        for item in tmp1:
            values.append(item) 
        
        indices = []
        for item in tmp2:
            indices.append(item)    
                              
        #else:
        #    values = self.N[0]
        #    indices = indices[0]     
        #    dcol = np.diff(values)[0]    
        
        self.indices = np.array(indices)
        self.axes = np.array(values)
        self.elements_per_table = np.prod(self.indices.shape)
            
        #return np.array(values), #Nd, Nt, dims, values, indices, columns, dcol, locs, colnames
            
    def DatasetName(self, integral, species, donor_species):
        """
        Return name of table.  Will be called from dictionary or hdf5 using
        this name.
        """    
        
        if integral in ['PhiWiggle', 'PsiWiggle']:
            return "log%s_%i%i" % (integral, species, donor_species)
        elif integral == 'Tau':
            return 'log%s' % integral    
        else:
            return "log%s_%s" % (integral, species)   
              
    def TabulateRateIntegrals(self, retabulate = True):
        """
        Return a dictionary of lookup tables, and also store a copy as self.itabs.
        """
        
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

                pb = ProgressBar(self.elements_per_table, name)
                    
                tab = np.zeros(self.dims)
                for j, ind in enumerate(self.indices):

                    if j % size != rank:
                        continue
                               
                    tab[ind] = \
                        self.Tabulate(integral, absorber, self.axes[j])
                    
                    pb.update(j)
                    
                tabs[name] = tab.copy()
                
            if re.search('Wiggle', name):
                if self.grid.metals:
                    if self.grid.absorbers[i_donor + 1] in self.grid.metal_ions:
                        h += 1
                    else:
                        i_donor += 1
                else:    
                    i_donor += 1 
            else:
                h += 1    

            pb.finish()
                                
        print 'Integral tabulation complete.'
        
        return tabs         
            
    def TotalOpticalDepth(self, N):
        """
        Optical depth due to all absorbing species at given column density.
        Assumes ncol is a 3-element array.
        """
        
        tau = 0.0
        for absorber in self.grid.absorbers:
            tau += self.OpticalDepth(10**N[self.grid.absorbers.index(absorber)], 
                absorber)
    
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
                max(self.grid.ioniz_thresholds[absorber], self.src.Emin), 
                self.src.Emax)[0]
        else:                                                                                                                                                                                
            result = np.sum(self.PartialOpticalDepth(self.src.E, ncol, species)[self.src.E > E_th[species]])
            
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
                    tmp += self.PartialOpticalDepth(energy, N[i], absorber)
            
            tau[j] = tmp     
            del tmp
        
        return tau
        
    def Tabulate(self, integral, absorber, N):
        if integral == 'Phi':
            table = self.Phi(absorber, N)
        if integral == 'Psi':
            table = self.Psi(absorber, N)
        if integral == 'Tau':
            table = self.TotalOpticalDepth(N)
            
        return np.log10(table)
        
    def Phi(self, absorber, N):
        """
        Equation 10 in Mirocha et al. 2012.
        """
                                                         
        # Otherwise, continuous spectrum                
        if self.pf['photon_conserving']:
            #if self.pf['SpectrumFile'] is not 'None':
            #    return np.trapz(self.src.Spectrum(None, t = t)[self.src.i_Eth[species]:] * \
            #        np.exp(-self.SpecificOpticalDepth(self.src.E[self.src.i_Eth[species]:], ncol)) / \
            #        (self.src.E[self.src.i_Eth[species]:] * erg_per_ev), self.src.E[self.src.i_Eth[species]:])
            #else:
            t = 0
            integrand = lambda E: self.src.Spectrum(E, t = t) * \
                np.exp(-self.SpecificOpticalDepth(E, 10**N)[0]) / E
                            
        else:
            integrand = lambda E: 1e-10 * PhotoIonizationCrossSection(E, species) * \
                self.src.Spectrum(E, t = t) * \
                np.exp(-self.SpecificOpticalDepth(E, ncol)[0]) / \
                (E * erg_per_ev)
            
        return quad(integrand,
            max(self.grid.ioniz_thresholds[absorber], self.src.Emin), 
            self.src.Emax)[0] / erg_per_ev
        
    def Psi(self, absorber, N, species = 0, donor_species = 0, xHII = 0.0, t = None):            
        """
        Equation 11 in Mirocha et al. 2012.
        """        
        
        # Otherwise, continuous spectrum    
        if self.pf['photon_conserving']:
        #    if self.pf['SpectrumFile'] is not 'None':
        #        return np.trapz(self.src.Spectrum(t = t)[self.src.i_Eth[species]:] * \
        #            np.exp(-self.SpecificOpticalDepth(self.src.E[self.src.i_Eth[species]:], 
        #            ncol)), self.src.E[self.src.i_Eth[species]:])
            #else:
            integrand = lambda E: self.src.Spectrum(E, t = t) * \
                np.exp(-self.SpecificOpticalDepth(E, 10**N)[0])
        else:
            integrand = lambda E: PhotoIonizationCrossSection(E, species) * \
                self.src.Spectrum(E, t = t) * \
                np.exp(-self.SpecificOpticalDepth(E, ncol)[0])
        
        return quad(integrand, 
            max(self.grid.ioniz_thresholds[absorber], self.src.Emin), self.src.Emax)[0]
        
    def PhiWiggle(self, ncol, species = 0, donor_species = 0, xHII = 0.0, t = None):
        """
        Equation 2.18 in the manual.
        """        
        
        Ej = E_th[donor_species]
        
        # Otherwise, continuous spectrum                
        if self.pf['PhotonConserving']:
            integrand = lambda E: 1e10 * \
                self.esec.DepositionFraction(E - Ej, xHII, channel = species + 1) * \
                self.src.Spectrum(E, t = t) * \
                np.exp(-self.SpecificOpticalDepth(E, ncol)[0]) / \
                (E * erg_per_ev)
        else:
            integrand = lambda E: 1e10 * \
                self.esec.DepositionFraction(E, xHII, channel = species + 1) * \
                PhotoIonizationCrossSection(E, species) * \
                self.src.Spectrum(E, t = t) * \
                np.exp(-self.SpecificOpticalDepth(E, ncol)[0]) / \
                (E * erg_per_ev)
            
        c = self.esec.Energies >= max(Ej, self.src.Emin)
        c &= self.esec.Energies <= self.src.Emax
                        
        y = []
        for E in self.esec.Energies[c]:
            y.append(integrand(E))
             
        return 1e-10 * \
            np.trapz(np.array(y), self.esec.Energies[c])    
    
    def PsiWiggle(self, ncol, species = 0, donor_species = 0, xHII = 0.0, t = None):            
        """
        Equation 2.19 in the manual.
        """        
        
        Ej = E_th[donor_species]
        
        # Otherwise, continuous spectrum    
        if self.pf['PhotonConserving']:
            integrand = lambda E: 1e20 * \
                self.esec.DepositionFraction(E - Ej, xHII, channel = species + 1) * \
                self.src.Spectrum(E, t = t) * \
                np.exp(-self.SpecificOpticalDepth(E, ncol)[0])
        else:
            integrand = lambda E: 1e20 * PhotoIonizationCrossSection(E, species) * \
                self.src.Spectrum(E, t = t) * \
                np.exp(-self.SpecificOpticalDepth(E, ncol)[0])
                
        c = self.esec.Energies >= max(Ej, self.src.Emin)
        c &= self.esec.Energies <= self.src.Emax
                        
        y = []
        for E in self.esec.Energies[c]:
            y.append(integrand(E))
             
        return 1e-20 * \
            np.trapz(np.array(y), self.esec.Energies[c])          
                              
    def PhiHat(self, ncol, species = 0, donor_species = 0, xHII = 0.0, t = None):
        """
        Equation 2.20 in the manual.
        """        
        
        Ei = E_th[species]
        
        # Otherwise, continuous spectrum                
        if self.pf['PhotonConserving']:
            integrand = lambda E: 1e10 * \
                self.esec.DepositionFraction(E - Ei, xHII, channel = 0) * \
                self.src.Spectrum(E, t = t) * \
                np.exp(-self.SpecificOpticalDepth(E, ncol)[0]) / \
                (E * erg_per_ev)
        else:
            integrand = lambda E: 1e10 * \
                self.esec.DepositionFraction(E - Ei, xHII, channel = 0) * \
                PhotoIonizationCrossSection(E, species) * \
                self.src.Spectrum(E, t = t) * \
                np.exp(-self.SpecificOpticalDepth(E, ncol)[0]) / \
                (E * erg_per_ev)
        
        c = self.esec.Energies >= max(Ei, self.src.Emin)
        c &= self.esec.Energies <= self.src.Emax       
                                                
        y = []
        for E in self.esec.Energies[c]:
            y.append(integrand(E))
             
        return 1e-10 * \
            np.trapz(np.array(y), self.esec.Energies[c])          
                
    def PsiHat(self, ncol, species = 0, donor_species = 0, xHII = 0.0, t = None):            
        """
        Equation 2.21 in the manual.
        """        
        
        Ei = E_th[species]
        
        # Otherwise, continuous spectrum    
        if self.pf['PhotonConserving']:
            integrand = lambda E: 1e20 * \
                self.esec.DepositionFraction(E - Ei, xHII, channel = 0) * \
                self.src.Spectrum(E, t = t) * \
                np.exp(-self.SpecificOpticalDepth(E, ncol)[0])
        else:
            integrand = lambda E: 1e20 * \
                self.esec.DepositionFraction(E - Ei, xHII, channel = 0) * \
                PhotoIonizationCrossSection(E, species) * \
                self.src.Spectrum(E, t = t) * \
                np.exp(-self.SpecificOpticalDepth(E, ncol)[0])
        
        c = self.esec.Energies >= max(Ei, self.src.Emin)
        c &= self.esec.Energies <= self.src.Emax
                        
        y = []
        for E in self.esec.Energies[c]:
            y.append(integrand(E))
             
        return 1e-20 * \
            np.trapz(np.array(y), self.esec.Energies[c])  
            
    def PsiBreve(self, ncol, species = 0, donor_species = 0, x_HII = 0.0, t = None):
        """
        Return fractional Lyman-alpha excitation.
        """         
        
        pass
                                  
                    
            