"""

RadiationSources.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sun Jul 22 16:22:12 2012

Description: Container for several RadiationSource____ instances.  Will loop
over said instances in ionization and heating rate calculations.

"""

import sys
from scipy.integrate import quad
from .Interpolate import Interpolate
from .ComputeCrossSections import *
from .RadiationSourceFromFile import *
from .RadiationSourceIdealized import *
from .InitializeIntegralTables import *
from .ReadParameterFile import ReadParameterFile
from .SetDefaultParameterValues import SetDefaultSourceParameters

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    print "Module mpi4py not found.  No worries, we'll just run in serial."
    rank = 0
    size = 1  
    
E_th = [13.6, 24.6, 54.4]

class RadiationSources:
    def __init__(self, pf, retabulate = True, path = '.'):
        self.pf = pf
        self.Ns = int(pf['NumberOfSources'])
        self.path = path    # Location of all output
        
        self.all_sources = self.initialize_sources(retabulate)
        
        if self.pf['ExitAfterIntegralTabulation']:
            sys.exit('Integral tabulation complete.')             
    
    def initialize_sources(self, retabulate = True):
        """
        Construct list of RadiationSource____ class instances.
        """    
        
        sources = []
        for i in xrange(self.Ns):
                        
            if self.pf['SourceFiles'] != 'None':
                sf = self.pf.copy()
                src = SetDefaultSourceParameters()
                
                if self.path != '.':
                    output = '%s/%s' % (self.path, self.pf['SourceFiles'][i])
                else:
                    output = self.pf['SourceFiles'][i]
                
                tmp = ReadParameterFile(output, setdefs = False)
                src.update(tmp)
                sf.update(src)    
                if (sf['SpectrumFile'] != 'None') and (self.path != '.'):
                    sf['SpectrumFile'] = '%s/%s' % (self.path, sf['SpectrumFile'])
                            
            else:
                sf = self.pf
                        
            # Create RadiationSource class instance
            if sf['SpectrumFile'] != 'None':
                rs = RadiationSourceFromFile(sf)
            else:
                rs = RadiationSourceIdealized(sf)
                                
            # Initialize class
            rs.initialize()   
            
            # Initialize integral tables
            rs.iits = InitializeIntegralTables(sf, rs)
                        
            if self.path is not '.':
                sf['IntegralTable'] = '%s/%s' % (self.path, rs.iits.tname)
                rs.iits = InitializeIntegralTables(sf, rs)
            
            if not sf['DiscreteSpectrum'] or sf['ForceIntegralTabulation']:
                rs.itabs = rs.iits.TabulateRateIntegrals(retabulate = retabulate)
                rs.Interpolate = Interpolate(sf, rs.iits)
            else:
                rs.itabs = rs.Interpolate = None
            
            rs.TableAvailable = (not rs.pf['DiscreteSpectrum']) or (rs.pf['ForceIntegralTabulation'])
                
            # Add bandpass averaging stuff
            rs = self.frequency_averaged_quantities(rs)    
                
            sources.append(rs)    
                
        return sources
    
    def frequency_averaged_quantities(self, rs):
        """
        Take in RadiationSource____ instance, return with frequency averaged
        quantities (if parameter file says to do so).
        """    
        
        ### THESE ROUTINES BELONG IN RS CLASSES TheMSELVES - eh, doesn't really matter
        
        
        # Multi-group treatment
        if self.pf['FrequencyAveragedCrossSections']:
            
            if rs._name == 'RadiationSourceIdealized':
                Qnorm = rs.pf['SpectrumPhotonLuminosity'] / rs.Lbol / \
                    quad(lambda x: rs.Spectrum(x) / x, rs.EminNorm, rs.EmaxNorm)[0]
            else:
                Qnorm = np.trapz(rs.L_E / rs.E, rs.E)
            
            E = np.zeros(rs.pf['FrequencyGroups'])
            Qdot = np.zeros_like(E)
            bands = rs.pf['FrequencyBands']
            
            if len(bands) == (len(E) - 1):
                bands.append(rs.Emax)
            
            for i in xrange(int(rs.pf['FrequencyGroups'])):
                if rs._name == 'RadiationSourceIdealized':
                    L = quad(lambda x: rs.Spectrum(x), bands[i], bands[i + 1])[0]
                    Q = quad(lambda x: rs.Spectrum(x) / x, bands[i], bands[i + 1])[0]
                                
                    E[i] = L / Q
                    Qdot[i] = Qnorm * rs.Lbol * Q
                else:
                    i1 = np.argmin(np.abs(bands[i] - E))
                    i2 = np.argmin(np.abs(bands[i + 1] - E))
                    L = np.trapz(rs.L_E[i1:i2], rs.E[i1:i2])
                    Q = np.trapz(rs.L_E[i1:i2] / rs.E[i1:i2], rs.E[i1:i2])
                    
                    E[i] = L / Q
                    Qdot[i] = Qnorm * rs.Lbol * Q
        
            rs.E = E
            rs.Qdot = Qdot
            rs.bands = bands
        
        # Discretization techniques
        if rs.pf['DiscreteSpectrum'] and (not rs.pf['ForceIntegralTabulation']):
            
            # Multi-group method
            if rs.pf['FrequencyAveragedCrossSections']:
                rs.Nfg = int(pf['FrequencyGroups'])
                rs.sigma_bar = np.zeros([3, rs.Nfg])
                for i in xrange(3):
                    for j in xrange(rs.Nfg):
                        rs.sigma_bar[i] = EffectiveCrossSection(rs, rs.bands[j], self.rs.bands[j + 1], species = i)
                        
            # Polychromatic method                        
            else:
                rs.Nfg = len(rs.E)
                rs.sigma = np.zeros([3, rs.Nfg])
                for i in xrange(3):
                    rs.sigma[i] = PhotoIonizationCrossSection(rs.E, species = i)
                    
        # Optically thin approximations
        if rs.pf['AllowSmallTauApprox'] > 0:
            rs.sigma_bar = np.zeros(3)
            rs.sigma_wiggle = np.zeros(3)
            rs.hnu_bar = np.zeros(3)
            rs.bol_frac = np.zeros(3)
            for i in xrange(3):
                if i > 0 and (not self.pf['MultiSpecies']):
                    continue
                
                rs.sigma_bar[i] = EffectiveCrossSection(rs, E_th[i], rs.Emax, species = i)
                rs.sigma_wiggle[i] = EnergyWeightedCrossSection(rs, E_th[i], rs.Emax, species = i)
                rs.hnu_bar[i] = rs.FrequencyAveragedBin(species = i, Emin = E_th[i], Emax = rs.Emax)[0]
                
                if rs._name == 'RadiationSourceIdealized':
                    rs.bol_frac[i] = quad(rs.Spectrum, E_th[i], rs.Emax)[0]
                else:
                    i1 = np.argmin(np.abs(E_th[i] - E))
                    rs.bol_frac[i] = np.trapz(rs.L_E[i1:], rs.E[i1:])
        
            rs.Gamma_const = rs.sigma_bar * rs.bol_frac / rs.hnu_bar / erg_per_ev
            rs.Heat_const = erg_per_ev * \
                (rs.hnu_bar * rs.sigma_wiggle / rs.sigma_bar - E_th)               
                                                                                                            
            rs.gamma_const = np.zeros([3, 3])
            for i in xrange(3):
                rs.gamma_const[i] = ((rs.hnu_bar / E_th[i]) * \
                    (rs.sigma_wiggle / rs.sigma_bar) - \
                    (E_th / rs.hnu_bar[i]))             
    
        return rs    
                        
    def PlotCompositeSpectrum(self):
        """
        Loop over RadiationSource instances and construct a composite spectrum.
        """
        pass                    
            
    
        