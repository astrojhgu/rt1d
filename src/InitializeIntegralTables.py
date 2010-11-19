"""
InitializeIntegralTables.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-10-25.

Description: Tabulate integrals that appear in the rate equations.

To do:
    -Add progressbar.
     
"""

from ComputeCrossSections import *
from RadiationSource import *
from SecondaryElectrons import *
from scipy.integrate import romberg as integrate
import numpy as np
import h5py, os, re

E_HI = 13.6
E_HeI = 24.6
E_HeII = 54.4

m_e = 9.10938188*10**-28 		# Electron mass - [m_e] = g
m_p = 1.67262158*10**-24		# Proton mass - [m_p] = g
m_n = 1.67492729*10**-24        # Neutron mass - [m_n] = g

m_H = m_p + m_e
m_HeI = 2.0 * (m_p + m_n + m_e)
m_HeII = 2.0 * (m_p + m_n) + m_e

IntegralList = ['PhotoIonizationRateIntegralHI', 'ElectronHeatingIntegralHI', 'ElectronHeatingIntegralHeI', 'ElectronHeatingIntegralHeII', 'SecondaryIonizationRateIntegralHI']

class InitializeIntegralTables: 
    def __init__(self, pf, data):
        self.pf = pf
        self.rs = RadiationSource(pf)
        self.esec = SecondaryElectrons(pf)
        self.SourceSpectrum = pf["SourceSpectrum"]
        self.SourceDiscretization = pf["SourceDiscretization"]
        self.SourceSpectralEnergyBins = pf["SourceSpectralEnergyBins"]
        self.SourceTemperature = pf["SourceTemperature"]
        self.SourcePhotonLuminosity = pf["SourcePhotonLuminosity"]
        self.SourceMass = pf["SourceMass"]
        self.SourcePowerLawIndex = pf["SourcePowerLawIndex"]
        self.InitialRedshift = pf["InitialRedshift"]
        self.dx = pf["LengthUnits"] / pf["GridDimensions"]
        self.Emin = pf["SourceMinEnergy"]
        self.Emax = pf["SourceMaxEnergy"]
        self.MultiSpecies = pf["MultiSpecies"]
        self.HIColumnMin = 1e-5 * data["HIDensity"][0] * self.dx
        self.HIColumnMax = 10 * np.sum(data["HIDensity"] * self.dx)
        self.HINBins = pf["ColumnDensityBinsHI"]
        self.HeINBins = pf["ColumnDensityBinsHeI"]
        self.HeIINBins = pf["ColumnDensityBinsHeII"]
        self.HIColumn = np.logspace(np.log10(self.HIColumnMin), np.log10(self.HIColumnMax), self.HINBins)
        
        self.rt1d = os.environ.get("RT1D")
                
        if self.MultiSpecies > 0: 
            self.HeIColumnMin = 1e-5 * data["HIDensity"][0] * self.dx
            self.HeIColumnMax = 10 * np.sum(data["HeIDensity"] * self.dx)
            self.HeIIColumnMin = 1e-5 * data["HeIIDensity"][0] * self.dx
            self.HeIIColumnMax = 10 * np.sum(data["HeIIDensity"] * self.dx)
            self.HeIColumn = np.logspace(np.log10(self.HeIColumnMin), np.log10(self.HeIColumnMax), self.HeINBins)
            self.HeIIColumn = np.logspace(np.log10(self.HeIIColumnMin), np.log10(self.HeIIColumnMax), self.HeIINBins)
        else:
            self.HeIColumn = np.zeros_like(self.HIColumn)
            self.HeIIColumn = np.zeros_like(self.HIColumn)
                
        try: os.mkdir("tabs")
        except OSError:
            pass
        
    def DetermineTableName(self):    
        """
        Returns the filename following the convention:
                
            filename = SourceType_Source(Mass/Temperature)_{SourceSpectrum}_InitialRedshift.h5
                
            SourceType:     mono, bb, popIII, or bh
            SourceSpectrum: pl-alpha, agn, mcd (only applicable for bh sources)
        
        """
                
        zi = "z{0}".format(self.InitialRedshift)
        
        if self.MultiSpecies == 0: dim = "1D"
        else: dim = "3D"
        
        if self.SourceDiscretization == 0: cont = 'cont'
        else: cont = 'disc'
        
        if self.SourceSpectrum < 0: 
            src = "mono"
            mort = "{0:g}phot".format(int(self.SourcePhotonLuminosity))
            return "{0}_{1}_{2}_{3}_{4}.h5".format(src, mort, zi, dim, cont)
        
        if self.SourceSpectrum == 0: 
            src = "bb"
            mort = "{0}K".format(int(self.SourceTemperature))
            return "{0}_{1}_{2}_{3}_{4}.h5".format(src, mort, zi, dim, cont)
            
        elif self.SourceSpectrum == 1:
            src = "popIII"
            mort = "{0}M".format(int(self.SourceMass))
            return "{0}_{1}_{2}_{3}_{4}.h5".format(src, mort, zi, dim, cont)
            
        else: 
            src = "bh"
            mort = "{0}M".format(int(self.SourceMass))
            
            if self.SourceSpectrum == 2: spec = "pl-{0}".format(self.pf["SourcePowerLawIndex"])
            elif self.SourceSpectrum == 3: spec = "agn"
            elif self.SourceSpectrum == 4: spec = "mcd"
            else: spec = "unknown"
        
            return "{0}_{1}_{2}_{3}_{4}_{5}.h5".format(src, mort, spec, zi, dim, cont)
            
    def ReadIntegralTable(self):
        """
        Since the lookup tables for the integral values in the rate equations (equations 1-12 in 
        Thomas & Zaroubi 2007) can be tabulated given only properties of the source, it is ideal
        for parameter spaces to be for a given source.  However, it may be nice to compare different
        sources holding all else constant.  This routine will look for a preexisting hdf5 file
        with the lookup tables for the source we've specified.  Just need to determine a naming 
        convention for files that can uniquely identify the source properties.
        """
        
        filename = self.DetermineTableName()
        itab = {}
        
        if os.path.exists("{0}/input/{1}".format(self.rt1d, filename)): tabloc = "{0}/input/{1}".format(self.rt1d, filename)
        elif os.path.exists("tabs/{0}".format(filename)): tabloc = "tabs/{0}".format(filename)
        else:
            print "Did not find a pre-existing integral table.  Generating tabs/{0} now...\n".format(filename)
            return None
        
        print "Found an integral table for this source.  Reading tabs/{0}\n".format(filename)
        f = h5py.File("tabs/{0}".format(filename), 'r')
        
        for item in f["IntegralTable"]: itab[item] = f["IntegralTable"][item].value
        
        itab["HIColumnValues_x"] = f["ColumnVectors"]["HIColumnValues_x"].value
        
        if self.MultiSpecies > 0:
            itab["HeIColumnValues_y"] = f["ColumnVectors"]["HeIColumnValues_y"].value
            itab["HeIIColumnValues_z"] = f["ColumnVectors"]["HeIIColumnValues_z"].value
            
        return itab
                    
    def WriteIntegralTable(self, itabs):
        """
        Write out interpolation tables for the integrals that appear in our rate equations.
        """
        
        filename = self.DetermineTableName()                    
        f = h5py.File("tabs/{0}".format(filename), 'w') 

        pf_grp = f.create_group("ParameterFile")
        tab_grp = f.create_group("IntegralTable")
        col_grp = f.create_group("ColumnVectors")
        
        for par in self.pf: pf_grp.create_dataset(par, data = self.pf[par])
        for integral in itabs: tab_grp.create_dataset(integral, data = itabs[integral])
    
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

        if itabs is not None:
            return itabs
        else:
            itabs = {}     

            # If hydrogen-only
            if self.MultiSpecies == 0:
                for integral in IntegralList:
                    tab = np.zeros(self.HINBins)
                    if not re.search('HI', integral): continue
                    for i, ncol_HI in enumerate(self.HIColumn):
                        tab[i] = eval("self.{0}({1})".format(integral, [ncol_HI, 0.0, 0.0]))
                        
                    itabs[integral] = tab
                    del tab
                    
            # If we're including helium as well         
            else:
                for integral in IntegralList:
                    tab = np.zeros([self.HINBins, self.HeINBins, self.HeIINBins])
                    for i, ncol_HI in enumerate(self.HIColumn):
                        for j, ncol_HeI in enumerate(self.HeIColumn):
                            for k, ncol_HeII in enumerate(self.HeIIColumn):
                                tab[i][j][k] = eval("self.{0}({1})".format(integral, [ncol_HI, ncol_HeI, ncol_HeII]))    
                                    
                    itabs[integral] = tab
                    del tab
                    
            self.WriteIntegralTable(itabs)    
            return itabs
    
    def OpticalDepth(self, E, n):
        """
        Returns the optical depth at energy E due to column densities of HI, HeI, and HeII, which
        are stored in the variable 'n' as a three element array.
        """
        
        tau = 0.0
        for i, column in enumerate(n):
            tau += PhotoIonizationCrossSection(E, i) * column
                                                                                                
        return tau
        
    def ElectronHeatingIntegralHI(self, n = [0.0, 0.0, 0.0]):
        """
        Returns the amount of heat deposited by secondary electrons from HI ionizations.  This is 
        the first term in TZ07 Eq. 12.
        
            units: cm^2 / s
            notes: the dimensionality is resolved later when we multiply by the bolometric luminosity (erg / s),
                   the number density of collision partners (cm^-3), and divide by 4 pi r^2 (cm^-2), leaving us 
                   with a true heating rate in erg / cm^3 / s.
        """    
        
        if self.SourceDiscretization == 0:
            integrand = lambda E: PhotoIonizationCrossSection(E, 0) * (E - E_HI) * self.rs.Spectrum(E) * \
                np.exp(-self.OpticalDepth(E, n)) / E
                                                        
            integral = integrate(integrand, E_HI, self.Emax)
            
        else:
            integral = 0
            for E in self.SourceSpectralEnergyBins:
                integral += PhotoIonizationCrossSection(E, 0) * (E - E_HI) * self.rs.Spectrum(E) * \
                    np.exp(-self.OpticalDepth(E, n)) / E
                        
        return integral
            
    def ElectronHeatingIntegralHeI(self, n = [0.0, 0.0, 0.0]):
        """
        Returns the amount of heat deposited by secondary electrons from HeI ionizations.  For full explanation, 
        see 'ElectronHeatingIntegralHI'.
        """    
        
        integrand = lambda E: PhotoIonizationCrossSection(E, 1) * (E - E_HeI) * self.rs.Spectrum(E) * \
            np.exp(-self.OpticalDepth(E, n)) / E
            
        integral = integrate(integrand, E_HeI, self.Emax)
        
        else:
            integral = 0
            for E in self.SourceSpectralEnergyBins:
                integral += PhotoIonizationCrossSection(E, 1) * (E - E_HeI) * self.rs.Spectrum(E) * \
                    np.exp(-self.OpticalDepth(E, n)) / E
                                                               
        return integral
            
    def ElectronHeatingIntegralHeII(self, n = [0.0, 0.0, 0.0]):
        """
        Returns the amount of heat deposited by secondary electrons from HeII ionizations.  For full explanation, 
        see 'ElectronHeatingIntegralHI'.
        """   
        
        integrand = lambda E: PhotoIonizationCrossSection(E, 2) * (E - E_HeII) * self.rs.Spectrum(E) * \
            np.exp(-self.OpticalDepth(E, n)) / E   
            
        integral = integrate(integrand, E_HeII, self.Emax)
        
        else:
            integral = 0
            for E in self.SourceSpectralEnergyBins:
                integral += PhotoIonizationCrossSection(E, 2) * (E - E_HeII) * self.rs.Spectrum(E) * \
                    np.exp(-self.OpticalDepth(E, n)) / E
                                                               
        return integral    
    
    def PhotoIonizationRateIntegralHI(self, n = [0.0, 0.0, 0.0]):
        """
        Returns the value of the bound-free photoionization rate integral of HI.  However, because 
        source luminosities vary with time and distance, it is unnormalized.  To get a true 
        photoionization rate, one must multiply these values by the spectrum's normalization factor
        and divide by 4*np.pi*r^2. 
        """
        
        if self.SourceDiscretization == 0:
            integrand = lambda E: PhotoIonizationCrossSection(E, 0) * self.rs.Spectrum(E) * \
                np.exp(-self.OpticalDepth(E, n)) / (E * erg_per_ev)    
                
            integral = integrate(integrand, E_HI, self.Emax)
                        
        else:
            integral = 0
            for E in self.SourceSpectralEnergyBins:
                integral += PhotoIonizationCrossSection(E, 0) * self.rs.Spectrum(E) * \
                    np.exp(-self.OpticalDepth(E, n)) / (E * erg_per_ev)     
                                                                                                                                                                    
        return integral  
        
    def SecondaryIonizationRateIntegralHI(self, n = [0.0, 0.0, 0.0]):
        """
        
        """
        
        if self.SourceDiscretization == 0:
            integrand = lambda E: PhotoIonizationCrossSection(E, 0) * (E - E_HI) * self.rs.Spectrum(E) * \
                np.exp(-self.OpticalDepth(E, n)) / (E * erg_per_ev) / E_HI   
                
            integral = integrate(integrand, E_HI, self.Emax)
            
        else:
            integral = 0
            for E in self.SourceSpectralEnergyBins:
                integral += PhotoIonizationCrossSection(E, 0) * (E - E_HI) * self.rs.Spectrum(E) * \
                    np.exp(-self.OpticalDepth(E, n)) / (E * erg_per_ev) / E_HI 
                                                                                                      
        return integral  
        
    def HISecondaryIonizationRateHI(self):
        pass
        
    def HISecondaryIonizationRateHeI(self):
        pass
        
    def HeIPhotoIonizationRate(self):
        pass
        
    def HeISecondaryIonizationRateHI(self):
        pass
        
    def HeISecondaryIonizationRateHeI(self):
        pass
        
    def HeIIPhotoIonizationRate(self):
        pass
        
    
        
    