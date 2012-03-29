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
from rt1d.mods.Constants import *
from rt1d.mods.ComputeCrossSections import PhotoIonizationCrossSection
from rt1d.mods.RadiationSource import RadiationSource
from rt1d.mods.SecondaryElectrons import SecondaryElectrons

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
    from scipy.integrate import quad as integrate
except ImportError:
    print 'Module scipy not found.  Replacement integration routines are much slower :('
    from Integrate import simpson as integrate    

E_th = [13.6, 24.6, 54.4]

m_e = 9.10938188*10**-28 		# Electron mass - [m_e] = g
m_p = 1.67262158*10**-24		# Proton mass - [m_p] = g
m_n = 1.67492729*10**-24        # Neutron mass - [m_n] = g

m_H = m_p + m_e
m_HeI = 2.0 * (m_p + m_n + m_e)
m_HeII = 2.0 * (m_p + m_n) + m_e

tiny_number = 1e-30

IntegralList = ['PhotoIonizationRate', 'ElectronHeatingRate', 'TotalOpticalDepth']

class InitializeIntegralTables: 
    def __init__(self, pf, data, grid):
        self.pf = pf
        self.rs = RadiationSource(pf)
        self.esec = SecondaryElectrons(pf)
        
        self.OutputDirectory = pf["OutputDirectory"]
        self.ProgressBar = pf["ProgressBar"] and pb   
        self.ParallelizationMethod = pf["ParallelizationMethod"]
        self.IntegralTableName = pf["IntegralTableName"]
        self.RegenerateTable = pf["RegenerateTable"]
        
        # Physics, initial conditions, control parameters
        self.MultiSpecies = pf["MultiSpecies"]
        self.InitialRedshift = pf["InitialRedshift"]
        self.PhotonConserving = pf["PhotonConserving"]
        
        # Source parameters
        self.SourceType = pf["SourceType"]
        self.SourceTemperature = pf["SourceTemperature"]
        self.SourceMass = pf["SourceMass"]
        
        # Spectral parameters
        self.SpectrumPowerLawIndex = pf["SpectrumPowerLawIndex"]
        self.SpectrumPhotonLuminosity = pf["SpectrumPhotonLuminosity"]
        self.SpectrumMinEnergy = pf["SpectrumMinEnergy"]
        self.SpectrumMaxEnergy = pf["SpectrumMaxEnergy"]
        self.SpectrumAbsorbingColumn = pf["SpectrumAbsorbingColumn"]
        
        # Column densities - determine automatically
        self.n_H = data['HIDensity'] + data['HIIDensity']
        self.n_He = data['HeIDensity'] + data['HeIIDensity'] + data['HeIIIDensity']
        self.HIColumnMin = np.floor(np.log10(pf["MinimumSpeciesFraction"] * np.min(self.n_H * grid.dx)))
        self.HIColumnMax = np.ceil(np.log10(pf["LengthUnits"] * np.max(self.n_H)))
        self.HeIColumnMin = self.HeIIColumnMin = np.floor(np.log10(pf["MinimumSpeciesFraction"] * np.min(self.n_He * grid.dx)))
        self.HeIColumnMax = self.HeIIColumnMax = np.ceil(np.log10(pf["LengthUnits"] * np.max(self.n_He)))
        
        self.HINBins = pf["ColumnDensityBinsHI"]
        self.HeINBins = pf["ColumnDensityBinsHeI"]
        self.HeIINBins = pf["ColumnDensityBinsHeII"]
                        
        self.HIColumn = np.linspace(self.HIColumnMin, self.HIColumnMax, self.HINBins)

        # Set up column density vectors for each absorber
        if self.MultiSpecies > 0: 
            self.HeIColumn = np.linspace(self.HeIColumnMin, self.HeIColumnMax, self.HeINBins)
            self.HeIIColumn = np.linspace(self.HeIIColumnMin, self.HeIIColumnMax, self.HeIINBins)        
        else:
            self.HeIColumn = np.ones_like(self.HIColumn) * tiny_number
            self.HeIIColumn = np.ones_like(self.HIColumn) * tiny_number
                    
        self.AllColumns = [self.HIColumn, self.HeIColumn, self.HeIIColumn]                
        self.TableDims = np.array([len(self.HIColumn), len(self.HeIColumn), len(self.HeIIColumn)])
                              
        # Make output directory          
        try: 
            os.mkdir("{0}".format(self.OutputDirectory))
        except OSError: 
            pass
            
        # Retrive rt1d environment - look for tables in rt1d/input
        self.rt1d = os.environ.get("RT1D")

        if self.pf['DiscreteSpectrum']:
            self.zeros = np.zeros_like(self.rs.E)
        else:
            self.zeros = np.zeros(1)
            
        self.Lbol = self.rs.BolometricLuminosity(0.0)    
                
    def DetermineTableName(self):    
        """
        Returns the filename following the convention:
                
            filename = SourceType_<SourceTemperature>_{SourceType}.h5 # out of date
        
        """
        
        if self.IntegralTableName != 'None':
            return self.IntegralTableName
              
        ms = 'ms%i' % self.MultiSpecies
        pc = 'pc%i' % self.PhotonConserving        
        
        if self.rs.DiscreteSpectrum:
            sed = 'D'
        else:
            sed = 'C'
        
        if self.MultiSpecies:
            dims = '%ix%ix%i' % (self.HINBins, self.HeINBins, self.HeIINBins)
        else:
            dims = '%i' % self.HINBins
        
        if self.SourceType == 0: 
            src = "mf"
            prop = "{0:g}phot".format(int(self.SpectrumPhotonLuminosity))
        
        if self.SourceType == 1: 
            src = "bb"
            prop = "{0}K".format(int(self.SourceTemperature))
                                                              
        elif self.SourceType == 2:                            
            src = "popIII"                                    
            prop = "{0}M".format(int(self.SourceMass))        
            
        elif self.SourceType == 3:
            src = "pl"
            prop = "{0}".format(self.SpectrumPowerLawIndex)
        
        elif self.SourceType == 4:
            src = "apl"
            prop = "{0}_{0}n".format(self.SpectrumPowerLawIndex, self.SpectrumAbsorbingColumn)    
                    
        # Limits
        Hlim = '%i%i' % (self.HIColumn[0], self.HIColumn[-1])
        Helim = '%i%i' % (self.HeIColumn[0], self.HeIColumn[-1])        
                    
        return "{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}.h5".format(src, prop, pc, ms, sed, dims, Hlim, Helim)
            
    def DatasetName(self, integral, species, donor_species = None):
        """
        Return name of table (as stored in HDF5 file).
        """    
        
        if integral == 'SecondaryIonizationRate':
            return "%s%i%i" % (integral, species, donor_species)
        elif integral == 'TotalOpticalDepth':
            return integral
        else:
            return "%s%i" % (integral, species)        
            
    def ReadIntegralTable(self):
        """
        Since the lookup tables for the integral values in the rate equations (equations 1-12 in 
        Thomas & Zaroubi 2007) can be tabulated given only properties of the source, it is ideal
        for parameter spaces to be for a given source.  However, it may be nice to compare different
        sources holding all else constant.  This routine will look for a preexisting hdf5 file
        with the lookup tables for the source we've specified.  
        """
        
        filename = self.DetermineTableName()
        itab = {}
        
        table_from_pf = False                        
        if os.path.exists("{0}/input/{1}".format(self.rt1d, filename)): 
            tabloc = "{0}/input/{1}".format(self.rt1d, filename)
        elif os.path.exists("{0}/{1}".format(self.OutputDirectory, filename)): 
            tabloc = "{0}/{1}".format(self.OutputDirectory, filename)
        elif os.path.exists("%s" % self.IntegralTableName):
            tabloc = "%s" % self.IntegralTableName
            table_from_pf = True
        else:
            if rank == 0:
                print "\nDid not find a pre-existing integral table.  Generating {0}/{1} now...".format(self.OutputDirectory, filename)
                print "10^%g < ncol_HI < 10^%g" % (self.HIColumnMin, self.HIColumnMax)
                if self.MultiSpecies:
                    print "10^%g < ncol_HeI and ncol_HeII < 10^%g" % (self.HeIColumnMin, self.HeIColumnMax)
            return None
        
        if rank == 0 and table_from_pf:
            print "\nFound table supplied in parameter file.  Reading %s\n" % tabloc
        elif rank == 0:
            print "\nFound an integral table for this source.  Reading %s\n" % tabloc
        
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
            
            if self.RegenerateTable:
                if rank == 0:
                    print "Recreating now...\n"
                return None
            else:
                if rank == 0:
                    print "Set RegenerateTable = 1 to recreate this table.\n"    
        
        if self.MultiSpecies > 0:
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
            
                if self.RegenerateTable:
                    if rank == 0:
                        print "Recreating now...\n"
                    return None
                else:
                    if rank == 0:
                        print "Set RegenerateTable = 1 to recreate this table.\n"    
        
        # Override what's in parameter file if there is a preexisting table and
        # all the bounds are OK
        self.HIColumn = itab["HIColumnValues_x"]    
        
        if self.MultiSpecies > 0:
            self.HeIColumn = itab["HeIColumnValues_y"]
            self.HeIIColumn = itab["HeIIColumnValues_z"]
        
        return itab
                    
    def WriteIntegralTable(self, itabs):
        """
        Write out interpolation tables for the integrals that appear in our rate equations.
        """
        
        filename = self.DetermineTableName()                    
        f = h5py.File("{0}/{1}".format(self.OutputDirectory, filename), 'w') 

        pf_grp = f.create_group("ParameterFile")
        tab_grp = f.create_group("IntegralTable")
        col_grp = f.create_group("ColumnVectors")
        
        for par in self.pf: 
            pf_grp.create_dataset(par, data = self.pf[par])
        for integral in itabs: 
            tab_grp.create_dataset(integral, data = itabs[integral])
    
        col_grp.create_dataset("HIColumnValues_x", data = self.HIColumn)
        
        if self.MultiSpecies > 0:
            col_grp.create_dataset("HeIColumnValues_y", data = self.HeIColumn)
            col_grp.create_dataset("HeIIColumnValues_z", data = self.HeIIColumn)
        
        f.close()
                    
    def TabulateRateIntegrals(self):
        """
        Return a dictionary of lookup tables for the integrals in the rate equations we'll be solving.
        These integrals exist in equations (1) - (12) in Thomas & Zaroubi 2007.  For each integral, 
        we'll have at least a 3D data cube, where each element corresponds to the value of that integral
        given column densities for the absorbers: HI, HeI, and HeII.  If we choose to decompose the values
        of these integrals further by storing separately contributions from each part of the spectrum, 
        our lookup tables will be 4D (I think).
        
        Note: The luminosity of our sources may be time dependent, but as long as the spectra are time
        independent we are ok.  The luminosity is just a normalization so we can be pull it outside of
        the integrals (which are over energy).
        """
                
        itabs = self.ReadIntegralTable()

        # If there was a pre-existing table, return it
        if itabs is not None:
            return itabs
        
        # Otherwise, make a lookup table
        itabs = {}     
        
        # If hydrogen-only
        if self.MultiSpecies == 0:
                            
            # Loop over integrals                
            for h, integral in enumerate(IntegralList):
                
                name = self.DatasetName(integral, species = 0, donor_species = 0)
                
                # Print some info to the screen
                if rank == 0 and self.ParallelizationMethod == 1: 
                        print "\nComputing value of {0}{1}...".format(integral, 0)
                if rank == 0 and self.ProgressBar and self.ParallelizationMethod == 1: 
                        pbar = ProgressBar(widgets = widget, maxval = len(self.HIColumn)).start()
                                    
                # Loop over column density                    
                tab = np.zeros_like(self.HIColumn)
                for i, ncol_HI in enumerate(self.HIColumn):
                    
                    if self.ParallelizationMethod == 1 and (i % size != rank): 
                        continue
                    if rank == 0 and self.ProgressBar and self.ParallelizationMethod == 1:
                        pbar.update(i + 1)
                    
                    # Evaluate integral
                    tab[i] = eval("self.{0}({1}, 0)".format(integral, [10**ncol_HI, 1, 1]))
                
                if size > 1 and self.ParallelizationMethod == 1: 
                    tab = MPI.COMM_WORLD.allreduce(tab, tab)
        
                MPI.COMM_WORLD.barrier()
                if rank == 0 and self.ProgressBar and self.ParallelizationMethod == 1: 
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
                                
                                if self.ParallelizationMethod == 1 and (global_i % size != rank): 
                                    continue
                                
                                tab[i][j][k] = eval("self.{0}({1}, {2})".format(integral, [10**ncol_HI, 10**ncol_HeI, 10**ncol_HeII], species))  
                                                                
                                if rank == 0 and self.ProgressBar:
                                    pbar.update(global_i)                            
                    
                    if size > 1 and self.ParallelizationMethod == 1: 
                        tab = MPI.COMM_WORLD.allreduce(tab, tab)
                    
                    itabs[name] = np.log10(tab)
                    del tab
                    
                    MPI.COMM_WORLD.barrier()
                    if rank == 0 and self.ProgressBar and self.ParallelizationMethod == 1: 
                        pbar.finish() 
                        
        # Optical depths for individual species
        for i in xrange(3):
            
            if i > 0 and not self.MultiSpecies:
                continue
                
            if rank == 0: 
                print "\nComputing value of OpticalDepth%i..." % i
            if rank == 0:     
                pbar = ProgressBar(widgets = widget, maxval = self.TableDims[i]).start()
            
            tab = np.zeros(self.TableDims[i])
            for j, col in enumerate(self.AllColumns[i]): 
                        
                if self.ParallelizationMethod == 1 and (j % size != rank): 
                    continue
                
                tab[j] = self.OpticalDepth(10**col, species = i)
                
                if rank == 0 and self.ProgressBar:
                    pbar.update(j)   
            
            if size > 1 and self.ParallelizationMethod == 1: 
                tab = MPI.COMM_WORLD.allreduce(tab, tab)
            
            itabs['OpticalDepth%i' % i] = np.log10(tab) 
            del tab

        # Write-out data
        if rank == 0 or self.ParallelizationMethod == 2: 
            self.WriteIntegralTable(itabs)
            
        # Don't move on until root processor has written out data    
        if size > 1 and self.ParallelizationMethod == 1: 
            MPI.COMM_WORLD.barrier()       
            
        return itabs 
        
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
                
        if self.rs.DiscreteSpectrum == 0:
            integrand = lambda E: self.PartialOpticalDepth(E, ncol, species)                           
            result = integrate(integrand, max(self.rs.Emin, E_th[species]), self.rs.Emax, epsrel = 1e-8)[0]
                  
        else:                                                                                                                                                                                
            result = np.sum(self.PartialOpticalDepth(self.rs.E, ncol, species)[self.rs.E > E_th[species]])
            
        return result
        
    def PartialOpticalDepth(self, E, ncol, species = 0):
        """
        Returns the optical depth at energy E due to column density of species.
        """
                        
        return PhotoIonizationCrossSection(E, species) * ncol   
        
    def SpecificOpticalDepth(self, E, ncol):
        """
        Returns the optical depth at energy E due to column densities of HI, HeI, and HeII, which
        are stored in the variable 'ncol' as a three element array.
        """
                        
        if type(E) is float:
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
        
    def PhotoIonizationRate(self, ncol = [0.0, 0.0, 0.0], species = 0):
        """
        Returns the value of the bound-free photoionization rate integral of 'species'.  However, because 
        source luminosities vary with time and distance, it is unnormalized.  To get a true 
        photoionization rate, one must multiply these values by the spectrum's normalization factor. 
        """
        
        if self.rs.DiscreteSpectrum:
            if self.PhotonConserving:
                integral = self.rs.Spectrum(self.rs.E, Lbol = self.Lbol) * \
                    np.exp(-self.SpecificOpticalDepth(self.rs.E, ncol)) / \
                    (self.rs.E * erg_per_ev)
            else:
                integral = PhotoIonizationCrossSection(self.rs.E, species) * \
                    self.rs.Spectrum(self.rs.E, Lbol = self.Lbol) * \
                    np.exp(-self.SpecificOpticalDepth(self.rs.E, ncol)) / \
                    (self.rs.E * erg_per_ev) 
                
            return np.sum(integral)   
        
        # Otherwise, continuous spectrum                
        if self.PhotonConserving:
            integrand = lambda E: 1e10 * self.rs.Spectrum(E, Lbol = self.Lbol) * \
                np.exp(-self.SpecificOpticalDepth(E, ncol)[0]) / \
                (E * erg_per_ev)
        else:
            integrand = lambda E: 1e10 * PhotoIonizationCrossSection(E, species) * \
                self.rs.Spectrum(E, Lbol = self.Lbol) * \
                np.exp(-self.SpecificOpticalDepth(E, ncol)[0]) / \
                (E * erg_per_ev)
            
        return 1e-10 * integrate(integrand, max(E_th[species], self.rs.Emin), self.rs.Emax, epsrel = 1e-8)[0]
                    
    def ElectronHeatingRate(self, ncol = [0.0, 0.0, 0.0], species = 0):    
        """
        Returns the amount of heat deposited by photo-electrons from 'species'.  This is 
        the first term in TZ07 Eq. 12.
        
            units: cm^2 / s
            notes: the dimensionality is resolved later when we multiply by the bolometric luminosity (erg / s),
                   the number density of collision partners (cm^-3), and divide by 4 pi r^2 (cm^-2), leaving us 
                   with a true heating rate in erg / cm^3 / s.
        """    
        
        if self.rs.DiscreteSpectrum:
            if self.PhotonConserving:
                integral = self.rs.Spectrum(self.rs.E, Lbol = self.Lbol) * \
                    np.exp(-self.SpecificOpticalDepth(self.rs.E, ncol)) 
            else:
                integral = PhotoIonizationCrossSection(self.rs.E, species) * \
                    self.rs.Spectrum(self.rs.E, Lbol = self.Lbol) * \
                    np.exp(-self.SpecificOpticalDepth(self.rs.E, ncol))
                    
            return np.sum(integral)
            
        # Otherwise, continuous spectrum    
        if self.PhotonConserving:
            integrand = lambda E: 1e20 * self.rs.Spectrum(E, Lbol = self.Lbol) * \
                np.exp(-self.SpecificOpticalDepth(E, ncol)[0])
        else:
            integrand = lambda E: 1e20 * PhotoIonizationCrossSection(E, species) * \
                self.rs.Spectrum(E, Lbol = self.Lbol) * \
                np.exp(-self.SpecificOpticalDepth(E, ncol)[0])
        
        return 1e-20 * integrate(integrand, max(E_th[species], self.rs.Emin), self.rs.Emax, epsrel = 1e-8)[0]
        
            
                
                    
            