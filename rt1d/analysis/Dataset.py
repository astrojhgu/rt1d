"""
Dataset.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-12-01.

Description: General module for command line analysis of rt1d data.
    
Tutorial:
import rt1d.analysis as rta
ds = rta.Dataset('./pf.dat')
pl.loglog(ds.data[10].r / cm_per_kpc, ds[10].T)  # Plot the temperature profile!
     
"""

import os, re, h5py
import numpy as np
import pylab as pl
import rt1d.mods as rtm
from rt1d.analysis.DataDump import DataDump
        
class Dataset:
    def __init__(self, pf):
        """
        Initialize our analysis environment.  The variable 'dataset' can be either the master
        parameter file for a series of runs, or an individual parameter file for a single run.
        The parameter 'gd' is the global directory, which will default to the directory where
        we launched python from.
        
        Some jargon:
            'ds' = dataset, this is the dictionary we'll ultimately return
            'dsn' = dataset name, a string that is the name of the directory all datadumps live in
            'dd' = datadump, referring to the data in a specific time output in the entire dataset
            'ddf' = datadump file, just the filename of a particular dd, like 'dd0000.h5'
            
        Note: When supplying parameter file, if it is in the current directory, type './Filename' 
            so we get the path.
        
        """   
        
        # Global directory
        self.gd = os.getcwd()
                
        # Run directory
        self.rd = pf.rpartition('/')[0]        
                
        # Read in parameter file
        self.pf = rtm.ReadParameterFile(pf)
                        
        # Also need path to parameter file (not including the parameter file itself)
        self.od = self.pf["OutputDirectory"]
        
        self.data = self.load()
                                     
    def load(self):
        """
        Return object containing access to all datadumps for the run we've specified.
        """
        
        # List all data*dumps* in this data*set*.
        alldds = []
        for f in os.listdir("{0}/{1}/{2}".format(self.gd, self.rd, self.od)):
            if not re.search('.h5', f): continue
            alldds.append(f)
            
        ds = {}
        for ddf in alldds:
            f = h5py.File("{0}/{1}/{2}/{3}".format(self.gd, self.rd, self.od, ddf))
            ID = ddf.partition('.')[0].strip('dd')
            ds[int(ID)] = DataDump(f["Data"], f["ParameterFile"])
            f.close()
            
        return ds
            
    #def get_energy_deposition(self, dataset, pf, bin_size = 100, bin_spacing = 'lin'):
    #    """
    #    Return a dictionary where each entry is a 2D array representing the energy deposition rate (erg / s / cm^3)
    #    deposition by photons in each bin of the spectrum in a given channel (heat, ionization, etc.).
    #    
    #        bin_size (eV): size of energy bins for the spectrum
    #        dataset: only single time dump, i.e. sim[50]
    #        
    #        currently only doing photoionization
    #        dataset must contain column density information
    #    """
    #    
    #    t = dataset.t
    #    r = dataset.r
    #    rs = rm.RadiationSource(pf)
    #    
    #    n_bins = (pf["SpectrumMaxEnergy"] - pf["SpectrumMinEnergy"]) / bin_size + 1
    #    if bin_spacing == 'lin':
    #        bin_edges = np.linspace(pf["SpectrumMinEnergy"], pf["SpectrumMaxEnergy"], n_bins)
    #    elif bin_spacing == 'log':
    #        bin_edges = np.logspace(pf["SpectrumMinEnergy"], pf["SpectrumMaxEnergy"], n_bins)
    #    else:
    #        print 'Bin spacing not recognized!'
    #        return 0
    #     
    #    result = np.zeros([pf["GridDimensions"] - 1, n_bins], float) 
    #        
    #    n = zip(*[dataset.ncol_HI, dataset.ncol_HeI, dataset.ncol_HeII])   
    #    
    #    for i in np.arange(int(pf["GridDimensions"]) - 1):    
    #        for j, bin in enumerate(bin_edges):
    #            if bin == bin_edges[-1]: continue
    #            integrand = lambda E: PhotoIonizationCrossSection(E, 0) * rs.Spectrum(E) * \
    #                np.exp(-self.tau(E, n[i])) / (E * erg_per_ev)
    #            result[i][j] = rs.BolometricLuminosity(t) * integrate(integrand, bin, bin + bin_size)[0] \
    #                / 4. / np.pi / r[i]**2
    #                            
    #    return {'PhotoIonization': result}
    #                                
    #def tau(self, E, n):
    #    """
    #    Returns the optical depth at energy E due to column densities of HI, HeI, and HeII, which
    #    are stored in the variable 'n' as a three element array.  Transcribed from InitializeIntegralTables.py.
    #    """
    #    
    #    tau = 0.0
    #    for i, column in enumerate(n):
    #        tau += PhotoIonizationCrossSection(E, i) * column
    #                                                                                            
    #    return tau    
        
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            