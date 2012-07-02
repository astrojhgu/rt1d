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
import h5py, os, re, itertools
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
        if pf['CosmologicalExpansion']:
            self.HIColumnMin = np.floor(np.log10(pf['MinimumSpeciesFraction'] * self.cosm.nH0 * (1. + self.cosm.zf)**3 * min(self.grid.dx)))
            self.HIColumnMax = np.ceil(np.log10(self.cosm.nH0 * (1. + self.cosm.zi)**3 * pf['LengthUnits']))
            self.HeIColumnMin = self.HeIIColumnMin = np.floor(np.log10(10**self.HIColumnMin * self.cosm.y))
            self.HeIColumnMax = self.HeIIColumnMax = np.ceil(np.log10(10**self.HIColumnMax * self.cosm.y))
        else:    
            self.n_H = (1. - self.cosm.Y) * self.grid.density / m_H
            self.n_He = self.cosm.Y * self.grid.density / m_He
            self.HIColumnMin = np.floor(np.log10(pf['MinimumSpeciesFraction'] * np.min(self.n_H * self.grid.dx)))
            self.HIColumnMax = np.ceil(np.log10(pf["LengthUnits"] * np.max(self.n_H)))
            self.HeIColumnMin = self.HeIIColumnMin = np.floor(np.log10(pf['MinimumSpeciesFraction'] * np.min(self.n_He * self.grid.dx)))
            self.HeIColumnMax = self.HeIIColumnMax = np.ceil(np.log10(pf['LengthUnits'] * np.max(self.n_He)))            
        
        self.HINBins = pf['ColumnDensityBinsHI']
        self.HeINBins = pf['ColumnDensityBinsHeI']
        self.HeIINBins = pf['ColumnDensityBinsHeII']
                        
        self.HIColumn = np.linspace(self.HIColumnMin, self.HIColumnMax, self.HINBins)

        self.itabs = None

        # Set up column density vectors for each absorber
        if self.pf['MultiSpecies'] > 0: 
            self.HeIColumn = np.linspace(self.HeIColumnMin, self.HeIColumnMax, self.HeINBins)
            self.HeIIColumn = np.linspace(self.HeIIColumnMin, self.HeIIColumnMax, self.HeIINBins)        
        else:
            self.HeIColumn = np.ones_like(self.HIColumn) * tiny_number
            self.HeIIColumn = np.ones_like(self.HIColumn) * tiny_number
        
        # What quantities are we going to compute?
        self.IntegralList = self.ToCompute()
        
        # What will our table look like?
        self.Nd, self.Nt, self.dims, self.values, \
        self.indices, self.columns, self.dcolumns, self.locs = \
            self.TableProperties()
                        
        # Retrive rt1d environment - look for tables in rt1d/input
        self.rt1d = os.environ.get("RT1D")

        if pf['DiscreteSpectrum']:
            self.zeros = np.zeros_like(self.rs.E)
        else:
            self.zeros = np.zeros(1)
            
        self.tname = self.DetermineTableName()    
                                    
    def DetermineTableName(self):    
        """
        Returns the filename following the convention:
                
        filename = SourceType_UniqueSourceProperties_PhotonConserving_MultiSpecies_
            DiscreteOrContinuous_TimeDependent?_SecondaryIonization_TableDims_NHlimits_NHelimits.h5
        
        """
        
        if self.pf['IntegralTable'] != 'None':
            return self.pf['IntegralTable']
              
        ms = 'ms%i' % self.pf['MultiSpecies']
        pc = 'pc%i' % self.pf['PhotonConserving']
        si = 'si%i' % self.pf['SecondaryIonization']
        td = 'tdep%i' % self.pf['SourceTimeEvolution']
                
        if self.pf['DiscreteSpectrum']:
            sed = 'D'
        else:
            sed = 'C'
                
        dims = ''
        for i in self.dims:
            dims += '%ix' % i
        dims = dims.rstrip('x')    
        
        if self.pf['SourceType'] == 0: 
            src = "mf"
            prop = "{0:g}phot".format(int(self.pf['SpectrumPhotonLuminosity']))
        
        if self.pf['SourceType'] == 1: 
            src = "bb"
            prop = "T%g" % int(self.pf['SourceTemperature'])
                                                              
        elif self.pf['SourceType'] == 2:                            
            src = "popIII"                                    
            prop = "M%g" % int(self.pf['SourceMass'])
            
        elif self.pf['SourceType'] >= 3:
            src = 'bh_M%i' % int(self.pf['SourceMass'])
            prop = '' 
            if 3 in self.rs.SpectrumPars['Type']:
                src += 'mcd'
                prop += "f%g" % self.rs.SpectrumPars['Fraction'][self.rs.SpectrumPars['Type'].index(3)]
                if self.rs.SpectrumPars['AbsorbingColumn'][self.rs.SpectrumPars['Type'].index(3)] > 0:
                    prop += "_logN%g" % np.log10(self.rs.SpectrumPars['AbsorbingColumn'][self.rs.SpectrumPars['Type'].index(3)])
            if 4 in self.rs.SpectrumPars['Type']:
                src += 'pl' 
                if 3 in self.rs.SpectrumPars['Type']:
                    prop += '_'
                prop += "f%g" % self.rs.SpectrumPars['Fraction'][self.rs.SpectrumPars['Type'].index(4)]
                prop += "_in%g" % self.rs.SpectrumPars['PowerLawIndex'][self.rs.SpectrumPars['Type'].index(4)]
                if self.rs.SpectrumPars['AbsorbingColumn'][self.rs.SpectrumPars['Type'].index(4)] > 0:
                    prop += "_logN%g" % np.log10(self.rs.SpectrumPars['AbsorbingColumn'][self.rs.SpectrumPars['Type'].index(3)])

        # Limits
        Hlim = '%i%i' % (self.HIColumn[0], self.HIColumn[-1])
        Helim = '%i%i' % (self.HeIColumn[0], self.HeIColumn[-1])        
                    
        return "%s_%s_%s_%s_%s_%s_%s_%s_%s_%s.h5" % (src, prop, pc, ms, sed, td, si, dims, Hlim, Helim)
            
    def DatasetName(self, integral, species, donor_species = None):
        """
        Return name of table (as stored in HDF5 file).
        """    
        
        if integral in ['PhiWiggle', 'PsiWiggle']:
            return "log%s%i%i" % (integral, species, donor_species)
        elif integral == 'TotalOpticalDepth':
            return 'log%s' % integral    
        else:
            return "log%s%i" % (integral, species)   
            
    def SuitableTable(self):
        """
        There may be a table good enough for our purposes, but with slightly 
        different parameters.  Find it.
        
            i.e. SecondaryIonization = 0, 1 has no effect
            Resolution of table doesn't either.
        """          
        
        tname = self.DetermineTableName()
        ob, prop, pc, ms, cd, dt, si, dims, hlim, helim = tname.rstrip('.h5').split('_')
            
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
        elif os.path.exists("{0}/{1}".format(self.pf['OutputDirectory'], filename)): 
            tabloc = "{0}/{1}".format(self.pf['OutputDirectory'], filename)
        elif os.path.exists("%s" % self.pf['IntegralTable']):
            tabloc = "%s" % self.pf['IntegralTable']
            table_from_pf = True
        else:
            if rank == 0:
                print "Did not find a pre-existing integral table.  Generating {0}/{1} now...".format(self.pf['OutputDirectory'], filename)
                print "Range of hydrogen column densities: 10^%g < N_H / cm^-2 < 10^%g" % (self.HIColumnMin, self.HIColumnMax)
                if self.pf['MultiSpecies']:
                    print "Range of helium column densities: 10^%g < N_He / cm^-2 < 10^%g" % (self.HeIColumnMin, self.HeIColumnMax)
            return None
        
        if rank == 0 and table_from_pf:
            print "Found table supplied in parameter file.  Reading %s..." % tabloc
        elif rank == 0:
            print "Found an integral table for this source.  Reading %s..." % tabloc
        
        f = h5py.File("%s" % tabloc, 'r')
        
        for item in f["integrals"]: 
            itab[item] = f["integrals"][item].value
        
        itab["logNHI"] = f["columns"]["logNHI"].value
        if np.min(itab["logNHI"]) > self.HIColumnMin or \
            np.max(itab["logNHI"]) < self.HIColumnMax:
            
            if rank == 0:
                print "The hydrogen column bounds of the existing lookup table are inadequate for this simulation."
                print "We require: 10^%g < N_H / cm^-2 < 10^%g" % (self.HIColumnMin, self.HIColumnMax)
                print "            10^%g < N_He / cm^-2 < 10^%g" % (self.HeIColumnMin, self.HeIColumnMax)
            
            if self.pf['RegenerateTable']:
                if rank == 0:
                    print "Recreating now..."
                return None
            else:
                if rank == 0:
                    print "Set RegenerateTable = 1 to recreate this table."    
        
        if self.pf['MultiSpecies'] > 0:
            itab["logNHeI"] = f["columns"]["logNHeI"].value
            itab["logNHeII"] = f["columns"]["logNHeII"].value
        
            if np.min(itab["logNHeI"]) > self.HeIColumnMin or \
                np.max(itab["logNHeI"]) < self.HeIColumnMin or \
                np.min(itab["logNHeII"]) > self.HeIIColumnMin or \
                np.max(itab["logNHeII"]) < self.HeIIColumnMin:
                
                if rank == 0:
                    print "The helium column bounds of the existing lookup table are inadequate for this simulation."
                    print "We require: 10^%g < ncol_H < 10^%g" % (self.HIColumnMin, self.HIColumnMax)
                    print "            10^%g < ncol_He < 10^%g" % (self.HeIColumnMin, self.HeIColumnMax)
            
                if self.pf['RegenerateTable']:
                    if rank == 0:
                        print "Recreating now..."
                    return None
                else:
                    if rank == 0:
                        print "Set RegenerateTable = 1 to recreate this table."    
        
        # If SED time-dependent
        if self.pf['SourceTimeEvolution']:
            itab['Age'] = f["columns"]["Age"].value
            self.pf['AgeBins'] = len(itab['Age'])
        
        # Override what's in parameter file if there is a preexisting table and
        # all the bounds are OK
        self.HIColumn = itab["logNHI"]
        self.HINBins = len(itab["logNHI"])
        if self.pf['MultiSpecies'] > 0:
            self.HeIColumn = itab["logNHeI"]
            self.HeIIColumn = itab["logNHeII"]
            self.HeINBins = len(itab["logNHeI"])
            self.HeIINBins = len(itab["logNHeII"])
            
        # Re-run table properties    
        self.Nd, self.Nt, self.dims, self.values, \
        self.indices, self.columns, self.dcolumns, self.locs = \
            self.TableProperties() 
                    
        return itab
        
    def WriteIntegralTable(self, name, tab):
        """
        Write-out integral table to HDF5.
        """    
        
        if size > 1 and self.pf['ParallelizationMethod'] == 1 and rank > 0:
            pass
        else:  
            filename = self.tname 
            if hasattr(self, 'lookup_tab'):
                tab_grp = self.lookup_tab["integrals"]
            elif os.path.exists("%s/%s" % (self.pf['OutputDirectory'], filename)):
                self.lookup_tab = h5py.File("%s/%s" % (self.pf['OutputDirectory'], filename), 'a') 
                tab_grp = self.lookup_tab["integrals"]
            else:
                self.lookup_tab = h5py.File("%s/%s" % (self.pf['OutputDirectory'], filename), 'w') 
                pf_grp = self.lookup_tab.create_group("parameters")
                tab_grp = self.lookup_tab.create_group("integrals")
                col_grp = self.lookup_tab.create_group("columns")
            
                for par in self.pf: 
                    pf_grp.create_dataset(par, data = self.pf[par])
                    
                col_grp.create_dataset("logNHI", data = self.HIColumn)
            
                if self.pf['MultiSpecies'] > 0:
                    col_grp.create_dataset("logNHeI", data = self.HeIColumn)
                    col_grp.create_dataset("logNHeII", data = self.HeIIColumn)
                
                if self.pf['SecondaryIonization'] >= 2:
                    col_grp.create_dataset('logxHII', data = self.esec.log_xHII)
                
                if self.pf['SourceTimeEvolution']:
                    col_grp.create_dataset('Age', data = self.rs.Age)    
            
            tab_grp.create_dataset(name, data = tab)
        
        # Don't move on until root processor has written out data    
        if size > 1 and self.pf['ParallelizationMethod'] == 1: 
            MPI.COMM_WORLD.barrier()
                            
    def TabulateRateIntegrals(self):
        """
        Return a dictionary of lookup tables, and also store a copy as self.itabs.
        """
                
        itabs = self.ReadIntegralTable()

        # If there was a pre-existing table, return it
        # (assuming it has everything we need)
        items_missing = 0
        if itabs is not None:
            tnames = self.IntegralNames()
            has_keys = itabs.keys()
            for key in ['logNHI', 'logNHeI', 'logNHeII', 'logAge', 'logxHII']:
                if key in has_keys:
                    has_keys.remove(key)
                                
            for name in tnames:
                if name not in has_keys:
                    items_missing += 1
            
            if items_missing == 0:            
                self.itabs = itabs
                return itabs    
                
        # Otherwise, make a new lookup table
        else:
            has_keys = []
            itabs = {}     
         
        s = ''
        for i in self.dims:
            s += '%ix' % i
        s = s.rstrip('x')
                
        if rank == 0:        
            print '\nThis lookup table will contain %i unique %iD tables. Each will have %s (%i) elements.' % (self.Nt, self.Nd, s, np.prod(self.dims))
            print 'That is a grand total of %i table elements.' % (self.Nt * np.prod(self.dims))
            print 'If that sounds like a lot, you should consider hitting ctrl-C.'
            
            if items_missing > 0:
                print ' '
                    
        # Loop over integrals
        h = 0
        donor_species = 0  
        while h < len(self.IntegralList): 
            integral = self.IntegralList[h] 
                                                
            for species in np.arange(3):
                if species > 0 and not self.pf['MultiSpecies']:
                    continue                    
                
                name = self.DatasetName(integral, species = species, 
                    donor_species = donor_species)
                    
                if name in has_keys:
                    if rank == 0:
                        print 'Found table %s...' % name
                    continue   
                                    
                # Print some info to the screen
                if rank == 0 and self.pf['ParallelizationMethod'] == 1: 
                    print "\nComputing value of %s..." % name
                    if self.ProgressBar:
                        pbar = ProgressBar(widgets = widget, maxval = np.prod(self.dims)).start()                    
                                    
                tab = np.zeros(self.dims)
                for i, ind in enumerate(self.indices):
                    
                    if i % size != rank:
                        continue
                    
                    logNHI, logNHeI, logNHeII, logx, Age = self.parse_args(self.values[i]) 
                    N = 10**np.array([logNHI, logNHeI, logNHeII])
                    tab[ind] = eval('self.{0}([{1},{2},{3}], {4}, {5}, {6}, {7})'.format(integral, 
                        N[0], N[1], N[2], species, donor_species, 10**logx, Age))
                                                
                    if rank == 0 and self.ProgressBar and self.pf['ParallelizationMethod'] == 1:
                        pbar.update(i)    
                    
                if size > 1 and self.pf['ParallelizationMethod'] == 1: 
                    tab = MPI.COMM_WORLD.allreduce(tab, tab)
        
                if rank == 0 and self.ProgressBar and self.pf['ParallelizationMethod'] == 1: 
                    pbar.finish()    
                    
                # Store table
                itabs[name] = np.log10(tab)
                self.WriteIntegralTable(name, itabs[name])
                del tab   
                                
            # Increment/Decrement donor_species
            if re.search('Wiggle', name) and (donor_species < 2 and self.pf['MultiSpecies']):
                donor_species += 1 
            elif re.search('Wiggle', name) and donor_species == 2:
                donor_species = 0
                h += 1
            else:
                h += 1    
                
        # Optical depths for individual species
        for i in xrange(3):
            
            if i > 0 and not self.pf['MultiSpecies']:
                continue
            
            name = 'logOpticalDepth%i' % i
                
            if name in has_keys:
                if rank == 0:
                    print 'Found table %s...' % name
                continue       
                
            if rank == 0: 
                print "\nComputing value of %s..." % name
            if rank == 0 and self.ProgressBar:     
                pbar = ProgressBar(widgets = widget, maxval = self.dims[i]).start()
            
            tab = np.zeros(self.dims[i])
            for j, col in enumerate(self.columns[i]): 
                        
                if self.pf['ParallelizationMethod'] == 1 and (j % size != rank): 
                    continue
                
                tab[j] = self.OpticalDepth(10**col, species = i)
                
                if rank == 0 and self.ProgressBar:
                    pbar.update(j)   
            
            if size > 1 and self.pf['ParallelizationMethod'] == 1: 
                tab = MPI.COMM_WORLD.allreduce(tab, tab)

            if rank == 0 and self.ProgressBar and self.pf['ParallelizationMethod'] == 1: 
                pbar.finish()
            
            itabs[name] = np.log10(tab) 
            self.WriteIntegralTable(name, itabs[name])
            del tab   
            
        if rank == 0:    
            self.lookup_tab.close()    
            
        self.itabs = itabs    
        return itabs         
            
    def parse_args(self, values):
        """
        Figures out what's what.
        """        
                
        if self.Nd == 1:
            return values, 0, 0, 0, 0
        elif self.Nd == 2:
            if self.pf['SecondaryIonization'] < 2:
                return values[0], 0, 0, 0, values[1]
            else:
                return values[0], 0, 0, values[1], 0
        elif self.Nd == 3:
            if self.pf['MultiSpecies']:
                return values[0], values[1], values[2], 0, 0
            else:
                return values[0], 0, 0, values[1], values[2]
        elif self.Nd == 4:
            if self.pf['SecondaryIonization'] < 2:
                return values[0], values[1], values[2], 0, values[3]
            else:
                return values[0], values[1], values[2], values[3], 0
        else:
            return values
            
    def TableProperties(self):
        """
        Figure out ND space of all lookup table elements.
        """  
        
        Ns = 1. + 2 * self.pf['MultiSpecies']  # number of species      

        Nd = 1                  # Table dimensions
        Nt = 1. * Ns            # Number of tables (unique quantities)
        dims = [self.HINBins]   # Number of elements in each dimension of each table        
        columns = [self.HIColumn]                    
        locs = [0]
                                    
        if not self.pf['Isothermal']:
            Nt += 1 * Ns   
                                
        if self.pf['MultiSpecies'] >= 1:
            Nd += 2
            dims.extend([self.HeINBins, self.HeIINBins]) 
            columns.extend([self.HeIColumn, self.HeIIColumn]) 
            locs.extend([1, 2])
          
        if self.pf['SecondaryIonization'] >= 2:
            Nd += 1
            Nt += 2. * (1. + 8 * self.pf['MultiSpecies'])  # Phi/Psi Wiggle
            Nt += 2. * Ns                                  # Phi/Psi Hat
            dims.append(self.pf['IonizedFractionBins'])
            columns.append(self.esec.log_xHII)
            locs.append(3)
            
        if self.pf['SourceTimeEvolution']:
            Nd += 1
            dims.append(self.pf['AgeBins'])
            columns.append(self.rs.Age)
            locs.append(4)
            
        indices = []
        for element in columns:
            indices.append(np.arange(len(element)))
                
        if Nd > 1:
            tmp1 = itertools.product(*columns)
            tmp2 = itertools.product(*indices)
            
            values = []
            for item in tmp1:
                values.append(item) 
            
            indices = []
            for item in tmp2:
                indices.append(item)    
                
            # spacing of columns    
            dcol = []
            for col in columns:
                dcol.append(np.diff(col)[0])
                  
        else:
            values = columns[0]
            indices = indices[0]     
            dcol = np.diff(values)[0]    
            
        return Nd, Nt, dims, values, indices, columns, dcol, locs
                                     
    def ToCompute(self):
        """
        Return list of quantities to compute.
        """    

        integrals = ['Phi']
        if not self.pf['Isothermal']:
            integrals.append('Psi')
        
        if self.esec.Method >= 2:    
            integrals.extend(['PhiWiggle', 'PhiHat'])
            
            if not self.pf['Isothermal']:
                integrals.extend(['PsiWiggle', 'PsiHat'])
        
        return integrals
        
    def IntegralNames(self):
        """
        Full names including species indices.
        """    
        
        names = ['logOpticalDepth0']        
        integrals = self.ToCompute()
        
        for integral in integrals:
            if integral == 'Phi':
                names.append('logPhi0')
                if self.pf['MultiSpecies']:
                    names.extend(['logPhi1', 'logPhi2', 'logOpticalDepth1', 'logOpticalDepth2'])
            elif integral == 'Psi':
                names.append('logPsi0')
                if self.pf['MultiSpecies']:
                    names.extend(['logPsi1', 'logPsi2'])
            else:
                pass        
                
        if self.pf['SecondaryIonization'] >= 2:
            names.extend(['logPhiHat0', 'logPsiHat0'])
                        
            if self.pf['MultiSpecies']:
                names.extend(['logPhiHat1', 'logPsiHat1', 'logPhiHat2', 'logPsiHat2'])
                for i in xrange(3):
                    for j in xrange(3):
                        names.extend(['logPhiWiggle%i%i' % (i, j), 'logPsiWiggle%i%i' % (i, j)])
            else:
                names.extend(['logPhiWiggle00', 'logPsiWiggle00'])
            
        return names
            
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
                
        if self.pf['DiscreteSpectrum'] == 0:
            integrand = lambda E: self.PartialOpticalDepth(E, ncol, species)                           
            result = integrate(integrand, max(self.rs.Emin, E_th[species]), self.rs.Emax)[0]
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
        
    def Phi(self, ncol, species = 0, donor_species = 0, xHII = 0.0, t = None):
        """
        Equation 10 in Mirocha et al. 2012.
        """      
                                
        # Otherwise, continuous spectrum                
        if self.pf['PhotonConserving']:
            integrand = lambda E: 1e-10 * self.rs.Spectrum(E, t = t) * \
                np.exp(-self.SpecificOpticalDepth(E, ncol)[0]) / \
                (E * erg_per_ev)
        else:
            integrand = lambda E: 1e-10 * PhotoIonizationCrossSection(E, species) * \
                self.rs.Spectrum(E, t = t) * \
                np.exp(-self.SpecificOpticalDepth(E, ncol)[0]) / \
                (E * erg_per_ev)
            
        return 1e10 * integrate(integrand, max(E_th[species], self.rs.Emin), self.rs.Emax)[0]
        
    def Psi(self, ncol, species = 0, donor_species = 0, xHII = 0.0, t = None):            
        """
        Equation 11 in Mirocha et al. 2012.
        """        
        
        # Otherwise, continuous spectrum    
        if self.pf['PhotonConserving']:
            integrand = lambda E: self.rs.Spectrum(E, t = t) * \
                np.exp(-self.SpecificOpticalDepth(E, ncol)[0])
        else:
            integrand = lambda E: PhotoIonizationCrossSection(E, species) * \
                self.rs.Spectrum(E, t = t) * \
                np.exp(-self.SpecificOpticalDepth(E, ncol)[0])
        
        return integrate(integrand, max(E_th[species], self.rs.Emin), self.rs.Emax, epsrel = 1e-8)[0]
        
    def PhiWiggle(self, ncol, species = 0, donor_species = 0, xHII = 0.0, t = None):
        """
        Equation 2.18 in the manual.
        """        
        
        Ej = E_th[donor_species]
        
        # Otherwise, continuous spectrum                
        if self.pf['PhotonConserving']:
            integrand = lambda E: 1e10 * \
                self.esec.DepositionFraction(E - Ej, xHII, channel = species + 1) * \
                self.rs.Spectrum(E, t = t) * \
                np.exp(-self.SpecificOpticalDepth(E, ncol)[0]) / \
                (E * erg_per_ev)
        else:
            integrand = lambda E: 1e10 * \
                self.esec.DepositionFraction(E, xHII, channel = species + 1) * \
                PhotoIonizationCrossSection(E, species) * \
                self.rs.Spectrum(E, t = t) * \
                np.exp(-self.SpecificOpticalDepth(E, ncol)[0]) / \
                (E * erg_per_ev)
            
        c = self.esec.Energies >= max(Ej, self.rs.Emin)
        c &= self.esec.Energies <= self.rs.Emax
                        
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
                self.rs.Spectrum(E, t = t) * \
                np.exp(-self.SpecificOpticalDepth(E, ncol)[0])
        else:
            integrand = lambda E: 1e20 * PhotoIonizationCrossSection(E, species) * \
                self.rs.Spectrum(E, t = t) * \
                np.exp(-self.SpecificOpticalDepth(E, ncol)[0])
                
        c = self.esec.Energies >= max(Ej, self.rs.Emin)
        c &= self.esec.Energies <= self.rs.Emax
                        
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
                self.rs.Spectrum(E, t = t) * \
                np.exp(-self.SpecificOpticalDepth(E, ncol)[0]) / \
                (E * erg_per_ev)
        else:
            integrand = lambda E: 1e10 * \
                self.esec.DepositionFraction(E - Ei, xHII, channel = 0) * \
                PhotoIonizationCrossSection(E, species) * \
                self.rs.Spectrum(E, t = t) * \
                np.exp(-self.SpecificOpticalDepth(E, ncol)[0]) / \
                (E * erg_per_ev)
        
        c = self.esec.Energies >= max(Ei, self.rs.Emin)
        c &= self.esec.Energies <= self.rs.Emax       
                                                
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
                self.rs.Spectrum(E, t = t) * \
                np.exp(-self.SpecificOpticalDepth(E, ncol)[0])
        else:
            integrand = lambda E: 1e20 * \
                self.esec.DepositionFraction(E - Ei, xHII, channel = 0) * \
                PhotoIonizationCrossSection(E, species) * \
                self.rs.Spectrum(E, t = t) * \
                np.exp(-self.SpecificOpticalDepth(E, ncol)[0])
        
        c = self.esec.Energies >= max(Ei, self.rs.Emin)
        c &= self.esec.Energies <= self.rs.Emax
                        
        y = []
        for E in self.esec.Energies[c]:
            y.append(integrand(E))
             
        return 1e-20 * \
            np.trapz(np.array(y), self.esec.Energies[c])   
                                  
                    
            