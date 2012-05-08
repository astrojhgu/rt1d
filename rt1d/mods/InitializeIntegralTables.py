"""
InitializeIntegralTables.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-10-25.

Description: Tabulate integrals that appear in the rate equations.

To do:
    -Add progressbar, allow integral tabulation in parallel.
     
"""

import numpy as np
import h5py, os, re
from .Constants import *
from .Cosmology import Cosmology 
from .InitializeGrid import InitializeGrid
from .RadiationSource import RadiationSource
from .SecondaryElectrons import SecondaryElectrons
from .ComputeCrossSections import PhotoIonizationCrossSection

try:
    from progressbar import *
    pb = True
    widget = ["rt1d: ", Percentage(), ' ', Bar(marker = RotatingMarker()), ' ', ETA(), ' ']
except ImportError:
    "Module progressbar not found."
    pb = False

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    print "Module mpi4py not found.  No worries, we'll just run in serial."
    rank = 0
    size = 1

try:
    import scipy
    from scipy.integrate import quad as integrate
except ImportError:
    print 'Module scipy not found.  Replacement integration routines are much slower :('
    from Integrate import simpson as integrate    

E_th = [13.6, 24.6, 54.4]

tiny_number = 1e-30
negligible_column = 1

scipy.seterr(all = 'ignore')

class InitializeIntegralTables: 
    def __init__(self, pf):
        self.pf = pf
        self.rs = RadiationSource(pf)
        self.cosm = Cosmology(pf)
        self.esec = SecondaryElectrons(pf)
        self.grid = InitializeGrid(pf)
                        
        self.ProgressBar = pf["ProgressBar"] and pb   

        # Column densities - determine automatically
        if pf.CosmologicalExpansion:
            self.HIColumnMin = np.floor(np.log10(pf.MinimumSpeciesFraction * self.cosm.nH0 * (1. + self.cosm.zf)**3 * self.grid.dx))
            self.HIColumnMax = np.ceil(np.log10(self.cosm.nH0 * (1. + self.cosm.zi)**3 * pf.LengthUnits))
            self.HeIColumnMin = self.HeIIColumnMin = np.floor(np.log10(10**self.HIColumnMin * self.cosm.y))
            self.HeIColumnMax = self.HeIIColumnMax = np.ceil(np.log10(10**self.HIColumnMax * self.cosm.y))
        else:    
            self.n_H = (1. - self.cosm.Y) * self.grid.density / m_H
            self.n_He = self.cosm.Y * self.grid.density / m_He
            self.HIColumnMin = np.floor(np.log10(pf.MinimumSpeciesFraction * np.min(self.n_H * self.grid.dx)))
            self.HIColumnMax = np.ceil(np.log10(pf["LengthUnits"] * np.max(self.n_H)))
            self.HeIColumnMin = self.HeIIColumnMin = np.floor(np.log10(pf.MinimumSpeciesFraction * np.min(self.n_He * self.grid.dx)))
            self.HeIColumnMax = self.HeIIColumnMax = np.ceil(np.log10(pf.LengthUnits * np.max(self.n_He)))            
        
        self.HINBins = pf.ColumnDensityBinsHI
        self.HeINBins = pf.ColumnDensityBinsHeI
        self.HeIINBins = pf.ColumnDensityBinsHeII
                        
        self.HIColumn = np.linspace(self.HIColumnMin, self.HIColumnMax, self.HINBins)

        # Set up column density vectors for each absorber
        if self.pf.MultiSpecies > 0: 
            self.HeIColumn = np.linspace(self.HeIColumnMin, self.HeIColumnMax, self.HeINBins)
            self.HeIIColumn = np.linspace(self.HeIIColumnMin, self.HeIIColumnMax, self.HeIINBins)        
        else:
            self.HeIColumn = np.ones_like(self.HIColumn) * tiny_number
            self.HeIIColumn = np.ones_like(self.HIColumn) * tiny_number
        
        self.itabs = None            
        self.AllColumns = [self.HIColumn, self.HeIColumn, self.HeIIColumn]                
        self.TableDims = np.array([len(self.HIColumn), len(self.HeIColumn), len(self.HeIIColumn)])
            
        # Retrive rt1d environment - look for tables in rt1d/input
        self.rt1d = os.environ.get("RT1D")

        if pf.DiscreteSpectrum:
            self.zeros = np.zeros_like(self.rs.E)
        else:
            self.zeros = np.zeros(1)
            
        self.Lbol = self.rs.BolometricLuminosity(0.0)
                
    def DetermineTableName(self):    
        """
        Returns the filename following the convention:
                
        filename = SourceType_UniqueSourceProperties_PhotonConserving_MultiSpecies_
            DiscreteOrContinuous_TableDims_NHlimits_NHelimits.h5
        
        """
        
        if self.pf.IntegralTableName != 'None':
            return self.pf.IntegralTableName
              
        ms = 'ms%i' % self.pf.MultiSpecies
        pc = 'pc%i' % self.pf.PhotonConserving        
        
        if self.pf.DiscreteSpectrum:
            sed = 'D'
        else:
            sed = 'C'
        
        if self.pf.MultiSpecies:
            dims = '%ix%ix%i' % (self.HINBins, self.HeINBins, self.HeIINBins)
        else:
            dims = '%i' % self.HINBins
        
        if self.pf.SourceType == 0: 
            src = "mf"
            prop = "{0:g}phot".format(int(self.pf.SpectrumPhotonLuminosity))
        
        if self.pf.SourceType == 1: 
            src = "bb"
            prop = "T%g" % int(self.pf.SourceTemperature)
                                                              
        elif self.pf.SourceType == 2:                            
            src = "popIII"                                    
            prop = "M%g" % int(self.pf.SourceMass)
            
        elif self.pf.SourceType == 3:
            src = "pl"
            prop = "%g" % self.pf.SpectrumPowerLawIndex
        
        elif self.pf.SourceType == 4:
            src = "apl"
            prop = "N%g_in%g" % (round(np.log10(self.pf.SpectrumAbsorbingColumn), 2), self.pf.SpectrumPowerLawIndex)  
      
        elif self.pf.SourceType == 5:
            src = "mcdpl"
            prop = "df%g_in%g" % (self.pf.SpectrumDiskFraction, self.pf.SpectrumPowerLawIndex) 
      
        # Limits
        Hlim = '%i%i' % (self.HIColumn[0], self.HIColumn[-1])
        Helim = '%i%i' % (self.HeIColumn[0], self.HeIColumn[-1])        
                    
        return "%s_%s_%s_%s_%s_%s_%s_%s.h5" % (src, prop, pc, ms, sed, dims, Hlim, Helim)
            
    def DatasetName(self, integral, species, donor_species = None):
        """
        Return name of table (as stored in HDF5 file).
        """    
        
        if integral in ['PhiWiggle', 'PsiWiggle']:
            return "%s%i%i" % (integral, species, donor_species)
        elif integral == 'TotalOpticalDepth':
            return integral    
        else:
            return "%s%i" % (integral, species)     
            
    def ReadIntegralTable(self):
        """
        Look for a preexisting hdf5 file with the lookup tables for the source 
        we've specified.  
        """
        
        filename = self.DetermineTableName()
        itab = {}
        
        # Check tables in rt1d/input directory, then other locations
        table_from_pf = False                        
        if os.path.exists("{0}/input/{1}".format(self.rt1d, filename)): 
            tabloc = "{0}/input/{1}".format(self.rt1d, filename)
        elif os.path.exists("{0}/{1}".format(self.pf.OutputDirectory, filename)): 
            tabloc = "{0}/{1}".format(self.pf.OutputDirectory, filename)
        elif os.path.exists("%s" % self.pf.IntegralTableName):
            tabloc = "%s" % self.pf.IntegralTableName
            table_from_pf = True
        else:
            if rank == 0:
                print "\nDid not find a pre-existing integral table.  Generating {0}/{1} now...".format(self.pf.OutputDirectory, filename)
                print "10^%g < ncol_HI < 10^%g" % (self.HIColumnMin, self.HIColumnMax)
                if self.pf.MultiSpecies:
                    print "10^%g < ncol_HeI and ncol_HeII < 10^%g" % (self.HeIColumnMin, self.HeIColumnMax)
            return None
        
        if rank == 0 and table_from_pf:
            print "\nFound table supplied in parameter file.  Reading %s" % tabloc
        elif rank == 0:
            print "\nFound an integral table for this source.  Reading %s" % tabloc
        
        f = h5py.File("%s" % tabloc, 'r')
        
        for item in f["IntegralTable"]: 
            itab[item] = f["IntegralTable"][item].value
        
        itab["HIColumnValues_x"] = f["ColumnVectors"]["HIColumnValues_x"].value
        if np.min(itab["HIColumnValues_x"]) > self.HIColumnMin or \
            np.max(itab["HIColumnValues_x"]) < self.HIColumnMax:
            
            if rank == 0:
                print "The hydrogen column bounds of the existing lookup table are inadequate for this simulation."
                print "We require: 10^%g < ncol_H < 10^%g" % (self.HIColumnMin, self.HIColumnMax)
                print "            10^%g < ncol_He < 10^%g" % (self.HeIColumnMin, self.HeIColumnMax)
            
            if self.pf.RegenerateTable:
                if rank == 0:
                    print "Recreating now..."
                return None
            else:
                if rank == 0:
                    print "Set RegenerateTable = 1 to recreate this table."    
        
        if self.pf.MultiSpecies > 0:
            itab["HeIColumnValues_y"] = f["ColumnVectors"]["HeIColumnValues_y"].value
            itab["HeIIColumnValues_z"] = f["ColumnVectors"]["HeIIColumnValues_z"].value
        
            if np.min(itab["HeIColumnValues_y"]) > self.HeIColumnMin or \
                np.max(itab["HeIColumnValues_y"]) < self.HeIColumnMin or \
                np.min(itab["HeIIColumnValues_z"]) > self.HeIIColumnMin or \
                np.max(itab["HeIIColumnValues_z"]) < self.HeIIColumnMin:
                
                if rank == 0:
                    print "The helium column bounds of the existing lookup table are inadequate for this simulation."
                    print "We require: 10^%g < ncol_H < 10^%g" % (self.HIColumnMin, self.HIColumnMax)
                    print "            10^%g < ncol_He < 10^%g" % (self.HeIColumnMin, self.HeIColumnMax)
            
                if self.pf.RegenerateTable:
                    if rank == 0:
                        print "Recreating now..."
                    return None
                else:
                    if rank == 0:
                        print "Set RegenerateTable = 1 to recreate this table."    
        
        # Override what's in parameter file if there is a preexisting table and
        # all the bounds are OK
        self.HIColumn = itab["HIColumnValues_x"]    
        if self.pf.MultiSpecies > 0:
            self.HeIColumn = itab["HeIColumnValues_y"]
            self.HeIIColumn = itab["HeIIColumnValues_z"]
        
        return itab
                    
    def WriteIntegralTable(self, itabs):
        """
        Write out interpolation tables.
        """
        
        filename = self.DetermineTableName()                    
        f = h5py.File("{0}/{1}".format(self.pf.OutputDirectory, filename), 'w') 

        pf_grp = f.create_group("ParameterFile")
        tab_grp = f.create_group("IntegralTable")
        col_grp = f.create_group("ColumnVectors")
        
        for par in self.pf: 
            pf_grp.create_dataset(par, data = self.pf[par])
        for integral in itabs: 
            tab_grp.create_dataset(integral, data = itabs[integral])
    
        col_grp.create_dataset("HIColumnValues_x", data = self.HIColumn)
        
        if self.pf.MultiSpecies > 0:
            col_grp.create_dataset("HeIColumnValues_y", data = self.HeIColumn)
            col_grp.create_dataset("HeIIColumnValues_z", data = self.HeIIColumn)
        
        f.close()
                    
    def TabulateRateIntegrals(self):
        """
        Return a dictionary of lookup tables, and also store a copy as self.itabs.
        """
                
        itabs = self.ReadIntegralTable()

        # If there was a pre-existing table, return it
        if itabs is not None:
            self.itabs = itabs
            return itabs
        
        # Otherwise, make a new lookup table
        itabs = {}     
        
        # What are we going to compute?
        IntegralList = self.ToCompute()
        
        # If hydrogen-only
        if self.pf.MultiSpecies == 0:
                            
            # Loop over integrals                
            for h, integral in enumerate(IntegralList):
                
                name = self.DatasetName(integral, species = 0, donor_species = 0)
                
                # Print some info to the screen
                if rank == 0 and self.pf.ParallelizationMethod == 1: 
                    print "\nComputing value of %s..." % name
                    if self.ProgressBar:
                        pbar = ProgressBar(widgets = widget, maxval = len(self.HIColumn)).start()
                
                if self.esec.Method < 2 or \
                    (('Wiggle' not in integral) and \
                     ('Hat' not in integral)):
                    tab = np.zeros_like(self.HIColumn)
                else:
                    tab = np.zeros([len(self.HIColumn), self.esec.NumberOfXiBins])
                                    
                # Loop over column density                                    
                for i, ncol_HI in enumerate(self.HIColumn):
                    
                    if self.pf.ParallelizationMethod == 1 and (i % size != rank): 
                        continue
                    if rank == 0 and self.ProgressBar and self.pf.ParallelizationMethod == 1:
                        pbar.update(i + 1)
                    
                    # Evaluate integral
                    if self.esec.Method < 2 or \
                        (('Wiggle' not in integral) and \
                         ('Hat' not in integral)):
                        tab[i] = eval("self.{0}({1}, 0)".format(integral, 
                            [10**ncol_HI, negligible_column, negligible_column]))
                    else:        
                        for j, xi in enumerate(self.esec.IonizedFractions):
                            tab[i][j] = eval("self.{0}({1}, 0, 0, {2})".format(integral, 
                            [10**ncol_HI, negligible_column, negligible_column], xi))
                
                if size > 1 and self.pf.ParallelizationMethod == 1: 
                    tab = MPI.COMM_WORLD.allreduce(tab, tab)
        
                MPI.COMM_WORLD.barrier()
                if rank == 0 and self.ProgressBar and self.pf.ParallelizationMethod == 1: 
                    pbar.finish()
        
                # Store table
                itabs[name] = np.log10(tab)
                del tab
                
        # If we're including helium as well         
        else:
                            
            for h, integral in enumerate(IntegralList):
                                                              
                for species in np.arange(3):
                    
                    if integral == 'TotalOpticalDepth' and species > 0:
                        continue
                    
                    name = self.DatasetName(integral, species = species)
                    
                    # This could take a while
                    if rank == 0: 
                        print "\nComputing value of %s..." % name
                    if rank == 0 and self.ProgressBar: 
                        pbar = ProgressBar(widgets = widget, maxval = np.prod(self.TableDims)).start()
                                        
                    # Loop over column densities
                    tab = np.zeros([len(self.HIColumn), len(self.HeIColumn), len(self.HeIIColumn)])
                    for i, ncol_HI in enumerate(self.HIColumn):                          
                        for j, ncol_HeI in enumerate(self.HeIColumn):
                            for k, ncol_HeII in enumerate(self.HeIIColumn):
                                
                                global_i = i * (self.TableDims[1] * self.TableDims[2]) + j * self.TableDims[2] + k + 1
                                
                                if self.pf.ParallelizationMethod == 1 and (global_i % size != rank): 
                                    continue
                                
                                tab[i][j][k] = eval("self.{0}({1}, {2})".format(integral, [10**ncol_HI, 10**ncol_HeI, 10**ncol_HeII], species))  
                                                                
                                if rank == 0 and self.ProgressBar:
                                    pbar.update(global_i)                            
                    
                    if size > 1 and self.pf.ParallelizationMethod == 1: 
                        tab = MPI.COMM_WORLD.allreduce(tab, tab)
                    
                    itabs[name] = np.log10(tab)
                    del tab
                    
                    MPI.COMM_WORLD.barrier()
                    if rank == 0 and self.ProgressBar and self.pf.ParallelizationMethod == 1: 
                        pbar.finish() 
                        
        # Optical depths for individual species
        for i in xrange(3):
            
            if i > 0 and not self.pf.MultiSpecies:
                continue
                
            if rank == 0: 
                print "\nComputing value of OpticalDepth%i..." % i
            if rank == 0 and self.ProgressBar:     
                pbar = ProgressBar(widgets = widget, maxval = self.TableDims[i]).start()
            
            tab = np.zeros(self.TableDims[i])
            for j, col in enumerate(self.AllColumns[i]): 
                        
                if self.pf.ParallelizationMethod == 1 and (j % size != rank): 
                    continue
                
                tab[j] = self.OpticalDepth(10**col, species = i)
                
                if rank == 0 and self.ProgressBar:
                    pbar.update(j)   
            
            if size > 1 and self.pf.ParallelizationMethod == 1: 
                tab = MPI.COMM_WORLD.allreduce(tab, tab)
            
            itabs['OpticalDepth%i' % i] = np.log10(tab) 
            del tab
            
            if rank == 0 and self.ProgressBar:
                pbar.finish()

        # Write-out data
        if rank == 0 or self.pf.ParallelizationMethod == 2: 
            self.WriteIntegralTable(itabs)
            
        # Don't move on until root processor has written out data    
        if size > 1 and self.pf.ParallelizationMethod == 1: 
            MPI.COMM_WORLD.barrier()       
            
        self.itabs = itabs    
        return itabs 
        
    def ToCompute(self):
        """
        Return list of quantities to compute.
        """    

        integrals = ['Phi', 'Psi']
        if self.esec.Method >= 2:    
            integrals.extend(['PhiWiggle', 'PsiWiggle', 'PhiHat', 'PsiHat'])
        
        return integrals
        
    def TotalOpticalDepth(self, ncol, species = None):
        """
        Optical depth due to all absorbing species at given column density.
        Assumes ncol is a 3-element array.
        """
    
        return self.OpticalDepth(ncol[0], 0) + self.OpticalDepth(ncol[1], 1) + \
            self.OpticalDepth(ncol[2], 2)
               
    def OpticalDepth(self, ncol, species = 0):
        """
        Optical depth of species integrated over entire spectrum at a 
        given column density.  We just use this to determine which cells
        are inside/outside of an I-front (OpticalDepthDefiningIfront = 0.5
        by default).
        """        
                
        if self.pf.DiscreteSpectrum == 0:
            integrand = lambda E: self.PartialOpticalDepth(E, ncol, species)                           
            result = integrate(integrand, max(self.rs.Emin, E_th[species]), self.rs.Emax, epsrel = 1e-8)[0]
                  
        else:                                                                                                                                                                                
            result = np.sum(self.PartialOpticalDepth(self.rs.E, ncol, species)[self.rs.E > E_th[species]])
            
        return result
        
    def PartialOpticalDepth(self, E, ncol, species = 0):
        """
        Returns the optical depth at energy E due to column density ncol of species.
        """
                        
        return PhotoIonizationCrossSection(E, species) * ncol   
        
    def SpecificOpticalDepth(self, E, ncol):
        """
        Returns the optical depth at energy E due to column densities of HI, HeI, and HeII, which
        are stored in the variable 'ncol' as a three element array.
        """
                                    
        if type(E) in [float, np.float32, np.float64]:
            E = [E]
                                           
        tau = self.zeros
        for i, energy in enumerate(E):
            tmp = 0
            
            if energy >= E_th[0]:
                tmp += self.PartialOpticalDepth(energy, ncol[0], 0)
            if energy >= E_th[1]:
                tmp += self.PartialOpticalDepth(energy, ncol[1], 1)
            if energy >= E_th[2]:
                tmp += self.PartialOpticalDepth(energy, ncol[2], 2)
                 
            tau[i] = tmp     
                        
        return tau
        
    def Phi(self, ncol, species = 0):
        """
        Equation 10 in Mirocha et al. 2012.
        """      
                
        # Otherwise, continuous spectrum                
        if self.pf.PhotonConserving:
            integrand = lambda E: 1e10 * self.rs.Spectrum(E, Lbol = self.Lbol) * \
                np.exp(-self.SpecificOpticalDepth(E, ncol)[0]) / \
                (E * erg_per_ev)
        else:
            integrand = lambda E: 1e10 * PhotoIonizationCrossSection(E, species) * \
                self.rs.Spectrum(E, Lbol = self.Lbol) * \
                np.exp(-self.SpecificOpticalDepth(E, ncol)[0]) / \
                (E * erg_per_ev)
            
        return 1e-10 * integrate(integrand, max(E_th[species], self.rs.Emin), self.rs.Emax, epsrel = 1e-8)[0]
        
    def Psi(self, ncol, species = 0):            
        """
        Equation 11 in Mirocha et al. 2012.
        """        
        
        # Otherwise, continuous spectrum    
        if self.pf.PhotonConserving:
            integrand = lambda E: 1e20 * self.rs.Spectrum(E, Lbol = self.Lbol) * \
                np.exp(-self.SpecificOpticalDepth(E, ncol)[0])
        else:
            integrand = lambda E: 1e20 * PhotoIonizationCrossSection(E, species) * \
                self.rs.Spectrum(E, Lbol = self.Lbol) * \
                np.exp(-self.SpecificOpticalDepth(E, ncol)[0])
        
        return 1e-20 * integrate(integrand, max(E_th[species], self.rs.Emin), self.rs.Emax, epsrel = 1e-8)[0]
        
    def PhiWiggle(self, ncol, species = 0, donor_species = 0, xHII = 0.0):
        """
        Equation 2.18 in the manual.
        """        
        
        Ej = E_th[donor_species]
        
        # Otherwise, continuous spectrum                
        if self.pf.PhotonConserving:
            integrand = lambda E: 1e10 * \
                self.esec.DepositionFraction(E - Ej, xHII, channel = species + 1) * \
                self.rs.Spectrum(E, Lbol = self.Lbol) * \
                np.exp(-self.SpecificOpticalDepth(E, ncol)[0]) / \
                (E * erg_per_ev)
        else:
            integrand = lambda E: 1e10 * \
                self.esec.DepositionFraction(E, xHII, channel = species + 1) * \
                PhotoIonizationCrossSection(E, species) * \
                self.rs.Spectrum(E, Lbol = self.Lbol) * \
                np.exp(-self.SpecificOpticalDepth(E, ncol)[0]) / \
                (E * erg_per_ev)
            
        c = self.esec.Energies >= max(Ej, self.rs.Emin)
        c &= self.esec.Energies <= self.rs.Emax
                        
        y = []
        for E in self.esec.Energies[c]:
            y.append(integrand(E))
             
        return 1e-10 * \
            np.trapz(np.array(y), self.esec.Energies[c])    
    
    def PsiWiggle(self, ncol, species = 0, donor_species = 0, xHII = 0.0):            
        """
        Equation 2.19 in the manual.
        """        
        
        Ej = E_th[donor_species]
        
        # Otherwise, continuous spectrum    
        if self.pf.PhotonConserving:
            integrand = lambda E: 1e20 * \
                self.esec.DepositionFraction(E - Ej, xHII, channel = species + 1) * \
                self.rs.Spectrum(E, Lbol = self.Lbol) * \
                np.exp(-self.SpecificOpticalDepth(E, ncol)[0])
        else:
            integrand = lambda E: 1e20 * PhotoIonizationCrossSection(E, species) * \
                self.rs.Spectrum(E, Lbol = self.Lbol) * \
                np.exp(-self.SpecificOpticalDepth(E, ncol)[0])
                
        c = self.esec.Energies >= max(Ej, self.rs.Emin)
        c &= self.esec.Energies <= self.rs.Emax
                        
        y = []
        for E in self.esec.Energies[c]:
            y.append(integrand(E))
             
        return 1e-20 * \
            np.trapz(np.array(y), self.esec.Energies[c])          
                              
    def PhiHat(self, ncol, species = 0, donor_species = None, xHII = 0.0):
        """
        Equation 2.20 in the manual.
        """        
        
        Ei = E_th[species]
        
        # Otherwise, continuous spectrum                
        if self.pf.PhotonConserving:
            integrand = lambda E: 1e10 * \
                self.esec.DepositionFraction(E - Ei, xHII, channel = 0) * \
                self.rs.Spectrum(E, Lbol = self.Lbol) * \
                np.exp(-self.SpecificOpticalDepth(E, ncol)[0]) / \
                (E * erg_per_ev)
        else:
            integrand = lambda E: 1e10 * \
                self.esec.DepositionFraction(E - Ei, xHII, channel = 0) * \
                PhotoIonizationCrossSection(E, species) * \
                self.rs.Spectrum(E, Lbol = self.Lbol) * \
                np.exp(-self.SpecificOpticalDepth(E, ncol)[0]) / \
                (E * erg_per_ev)
        
        c = self.esec.Energies >= max(Ei, self.rs.Emin)
        c &= self.esec.Energies <= self.rs.Emax       
                                                
        y = []
        for E in self.esec.Energies[c]:
            y.append(integrand(E))
             
        return 1e-10 * \
            np.trapz(np.array(y), self.esec.Energies[c])          
                
    def PsiHat(self, ncol, species = 0, donor_species = None, xHII = 0.0):            
        """
        Equation 2.21 in the manual.
        """        
        
        Ei = E_th[species]
        
        # Otherwise, continuous spectrum    
        if self.pf.PhotonConserving:
            integrand = lambda E: 1e20 * \
                self.esec.DepositionFraction(E - Ei, xHII, channel = 0) * \
                self.rs.Spectrum(E, Lbol = self.Lbol) * \
                np.exp(-self.SpecificOpticalDepth(E, ncol)[0])
        else:
            integrand = lambda E: 1e20 * \
                self.esec.DepositionFraction(E - Ei, xHII, channel = 0) * \
                PhotoIonizationCrossSection(E, species) * \
                self.rs.Spectrum(E, Lbol = self.Lbol) * \
                np.exp(-self.SpecificOpticalDepth(E, ncol)[0])
        
        c = self.esec.Energies >= max(Ei, self.rs.Emin)
        c &= self.esec.Energies <= self.rs.Emax
                        
        y = []
        for E in self.esec.Energies[c]:
            y.append(integrand(E))
             
        return 1e-20 * \
            np.trapz(np.array(y), self.esec.Energies[c])   
                                  
                    
            