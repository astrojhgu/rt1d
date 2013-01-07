"""

RadiationField.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Dec 31 11:16:59 2012

Description:

"""

import numpy as np
from ..util import parse_kwargs
from ..physics.SecondaryElectrons import *
from ..physics.Constants import erg_per_ev

class RadiationField:
    def __init__(self, grid, source, **kwargs):
        self.pf = parse_kwargs(**kwargs)
        self.grid = grid
        self.src = source
        self.esec = SecondaryElectrons(method = self.pf['secondary_ionization'])
        
        # Array of cross-sections to match grid size 
        if self.src.multi_freq:             
            self.sigma = (np.ones([self.grid.dims, self.src.Nfreq]) \
                       * self.src.sigma).T                  
                      
    def SourceDependentCoefficients(self, data, t):
        """
        Compute rate coefficients for photo-ionization, secondary ionization, 
        and photo-heating.
        """

        self.k_H = np.zeros_like(self.grid.zeros_grid_x_absorbers)
        self.Gamma = np.zeros_like(self.grid.zeros_grid_x_absorbers)
        self.gamma = np.zeros_like(self.grid.zeros_grid_x_absorbers2)
                
        if not self.src.SourceOn(t):
            return self.Gamma, self.gamma, self.k_H
        
        # Column density to cells (N) and of cells (Nc)    
        self.N, self.logN, self.Nc = self.grid.ColumnDensity(data)
         
        # Column densities (of all absorbers) sorted by cell 
        # (i.e. an array with shape = grid x # of absorbers)
        self.N_by_cell = np.zeros([self.grid.dims, len(self.grid.absorbers)])
        self.Nc_by_cell = np.zeros([self.grid.dims, len(self.grid.absorbers)])
        for i, absorber in enumerate(self.grid.absorbers):
            self.N_by_cell[...,i] = self.N[absorber]
            self.Nc_by_cell[...,i] = self.Nc[absorber]
        
        self.logN_by_cell = np.log10(self.N_by_cell)
        self.logNc_by_cell = np.log10(self.N_by_cell)
        
        # Number densities and normalizations
        self.n = {}
        self.A = {}
        for absorber in self.grid.absorbers:
            self.n[absorber] = data[absorber] * self.grid.x_to_n[absorber]
            self.A[absorber] = self.src.Lbol / self.n[absorber] / self.grid.Vsh
        
        # Eventually loop over sources here
        
        """
        For sources with discrete SEDs.
        """
        if self.src.discrete:
        
            # Loop over absorbing species
            for absorber in self.grid.absorbers:
                                
                # Discrete spectrum (multi-freq approach)
                if self.src.multi_freq:
                    self.MultiFreqCoefficients(data, absorber)
                
                # Discrete spectrum (multi-grp approach)
                elif self.src.multi_group:
                    pass
            
            return self.Gamma, self.gamma, self.k_H

        """
        For sources with continuous SEDs.
        """

        # Initialize some arrays
        self.PhiN = {}
        self.PhiNdN = {}
        if not self.pf['isothermal'] and self.pf['secondary_ionization'] < 2:
            self.PsiN = {}
            self.PsiNdN = {}
            self.fheat = self.esec.DepositionFraction(0.0, data['h_2'], 
                channel = 'heat') 
        
        if self.pf['secondary_ionization'] > 1:
            self.PhiWiggleN = {}
            self.PhiWiggleNdN = {}
            self.PhiHatN = {}
            self.PhiHatNdN = {}
            
            for absorber in self.grid.absorbers:
                self.PhiWiggleN[absorber] = {}
                self.PhiWiggleNdN[absorber] = {}
                self.PhiHatN[absorber] = {}
                self.PhiHatNdN[absorber] = {}
            
        else:
            self.fion = {}
            for absorber in self.grid.absorbers:
                self.fion[absorber] = self.esec.DepositionFraction(E = None, 
                    xHII = data['h_2'], channel = absorber)
        
        # Compute column densities up to and of cells        
        if self.pf['photon_conserving']:
            
            self.NdN = self.grid.N_absorbers \
                * [np.zeros_like(self.grid.zeros_grid_x_absorbers)]
            for i, absorber in enumerate(self.grid.absorbers):
                tmp = self.N_by_cell.copy()
                tmp[..., i] += self.Nc_by_cell[..., i]
                self.NdN[i] = tmp
                del tmp
            
            self.logNdN = np.log10(self.NdN)
        
        # Loop over absorbing species, compute tabulated quantities
        for i, absorber in enumerate(self.grid.absorbers):
               
            self.PhiN[absorber] = \
                10**self.src.tables["logPhi_%s" % absorber](self.logN_by_cell).squeeze()
            
            if (not self.pf['isothermal']) and (self.pf['secondary_ionization'] < 2):
                self.PsiN[absorber] = \
                    10**self.src.tables["logPsi_%s" % absorber](self.logN_by_cell).squeeze()
                
            if self.pf['photon_conserving']:
                self.PhiNdN[absorber] = \
                    10**self.src.tables["logPhi_%s" % absorber](self.logNdN[i]).squeeze()
                
                if (not self.pf['isothermal']) and (self.pf['secondary_ionization'] < 2):
                    self.PsiNdN[absorber] = \
                        10**self.src.tables["logPsi_%s" % absorber](self.logNdN[i]).squeeze()
        
            if self.pf['secondary_ionization'] > 1:
                
                if absorber in self.grid.metals:
                    continue
                
                for donor in self.grid.absorbers:
                    
                    suffix = '%s_%s' % (absorber, donor)
                         
                    self.PhiWiggleN[absorber][donor] = \
                        10**self.src.tables["logPhiWiggle_%s" % suffix](self.logN_by_cell).squeeze()    
                    self.PsiWiggleN[absorber][donor] = \
                        10**self.src.tables["logPsiWiggle_%s" % suffix](self.logN_by_cell).squeeze()
                    
                    if not self.pf['isothermal']:
                        self.PhiHatN[absorber] = \
                            10**self.src.tables["logPhiHat_%s" % suffix](self.logN_by_cell).squeeze()
                        self.PsiHatN[absorber] = \
                            10**self.src.tables["logPsiHat_%s" % suffix](self.logN_by_cell).squeeze()
                        
                    if not self.pf['photon_conserving']:
                        continue
                    
                    self.PhiWiggleNdN[absorber] = \
                        10**self.src.tables["logPhiWiggle_%s" % suffix](self.logNdN[j]).squeeze()    
                    self.PsiWiggleNdN[absorber] = \
                        10**self.src.tables["logPsiWiggle_%s" % suffix](self.logNdN[j]).squeeze()
                    
                    if not self.pf['isothermal']:
                        self.PhiHatNdN[absorber] = \
                            10**self.src.tables["logPhiHat_%s" % suffix](self.logNdN[i]).squeeze()
                        self.PsiHatNdN[absorber] = \
                            10**self.src.tables["logPsiHat_%s" % suffix](self.logNdN[i]).squeeze()
        
        # Now, go ahead and calculate the rate coefficients
        for i, absorber in enumerate(self.grid.absorbers):
            self.Gamma[..., i] = self.PhotoIonizationRate(absorber)
            self.k_H[..., i] = self.PhotoHeatingRate(absorber)
            
            if absorber in self.grid.metals:
                continue
            
            for j, donor in enumerate(self.grid.absorbers):
                self.gamma[..., i, j] = \
                    self.SecondaryIonizationRate(absorber, donor)
                   
        # Compute total optical depth too
        self.tau_tot = 10**self.src.tables["logTau"](self.logN_by_cell)            
        
        return self.Gamma, self.gamma, self.k_H
        
    def MultiFreqCoefficients(self, data, absorber):
        """
        Compute all source-dependent rates for given absorber assuming a
        multi-frequency SED.
        """
        
        i = self.grid.absorbers.index(absorber)
        n = self.n[absorber]
        N = self.N[absorber]
               
        # Optical depth up to cells at energy E
        N = np.ones([self.src.Nfreq, self.grid.dims]) * self.N[absorber]
        
        self.tau_r = N * self.sigma
        self.tau_tot = np.sum(self.tau_r, axis = 1)
                
        # Loop over energy groups
        self.Gamma_E = np.zeros([self.grid.dims, self.src.Nfreq])
        for j, E in enumerate(self.src.E):
            
            if E < self.grid.ioniz_thresholds[absorber]:
                continue    
            
            # Optical depth of cells (at this photon energy)                                                           
            tau_c = self.Nc[absorber] * self.src.sigma[j]
                                                            
            # Photo-ionization by *this* energy group
            self.Gamma_E[...,j] = \
                self.PhotoIonizationRateMultiFreq(self.src.Qdot[j], n, 
                self.tau_r[j], tau_c)
                          
            # Heating
            if self.grid.isothermal:
                continue
                 
            fheat = self.esec.DepositionFraction(E = E,
                xHII = data['h_2'], channel = 'heat')
            
            # Total energy deposition rate per atom i via photo-electrons 
            # due to ionizations by *this* energy group. 
            ee = self.Gamma_E[...,j] * (E - self.grid.ioniz_thresholds[absorber]) \
               * erg_per_ev 
            
            self.k_H[...,i] += ee * fheat
                
            if not self.pf['secondary_ionization']:
                continue
                                        
            # Ionizations of species k by photoelectrons from species i
            # Neglect HeII until somebody figures out how that works
            for k, otherabsorber in enumerate(self.grid.absorbers):
            
                # If these photo-electrons don't have enough 
                # energy to ionize species k, continue    
                if (E - self.grid.ioniz_thresholds[absorber]) < \
                    self.grid.ioniz_thresholds[otherabsorber]:
                    continue    
                
                fion = self.esec.DepositionFraction(E = E, 
                    xHII = data['h_2'], channel = absor1ber)
                                                                        
                # (This k) = i from paper, and (this i) = j from paper
                self.gamma[...,k,i] += ee * fion \
                    / (self.grid.ioniz_thresholds[otherabsorber] * erg_per_ev)
                                                                           
        # Total photo-ionization tally
        self.Gamma[...,i] = np.sum(self.Gamma_E, axis = 1)
    
    def PhotoIonizationRateMultiFreq(self, qdot, n, tau_r_E, tau_c):
        """
        Returns photo-ionization rate coefficient for single frequency over
        the entire grid.
        """     
                                        
        q0 = qdot * np.exp(-tau_r_E)             # number of photons entering cell per sec
        dq = q0 * (1. - np.exp(-tau_c))          # number of photons absorbed in cell per sec
        IonizationRate = dq / n / self.grid.Vsh  # ionizations / sec / atom        
                          
        return IonizationRate
        
    def PhotoIonizationRateMultiGroup(self):
        pass
        
    def PhotoIonizationRate(self, absorber):
        """
        Returns photo-ionization rate coefficient for continuous source.
        """                                     
            
        IonizationRate = self.PhiN[absorber].copy()
        
        if self.pf['photon_conserving']:
            IonizationRate -= self.PhiNdN[absorber]
        
        return self.A[absorber] * IonizationRate
        
    def PhotoHeatingRate(self, absorber):
        """
        Photo-electric heating rate coefficient due to photo-electrons previously 
        bound to `species.'  If this method is called, it means TabulateIntegrals = 1.
        """

        if self.esec.Method < 2:
            HeatingRate = self.PsiN[absorber].copy()
            HeatingRate -= self.grid.ioniz_thresholds[absorber] * erg_per_ev  \
                * self.PhiN[absorber]
            if self.pf['photon_conserving']:
                HeatingRate -= self.PsiNdN[absorber]
                HeatingRate += erg_per_ev \
                    * self.grid.ioniz_thresholds[absorber] \
                    * self.PhiNdN[absorber]
                    
        else:
            HeatingRate = self.PsiHatN[absorber].copy()
            HeatingRate -= self.grid.ioniz_thresholds[absorber] * erg_per_ev  \
                * self.PhiHatN[absorber]
            if self.pf['photon_conserving']:
                HeatingRate -= self.PsiHatNdN[absorber]
                HeatingRate += erg_per_ev \
                    * self.grid.ioniz_thresholds[absorber] \
                    * self.PhiHatNdN[absorber]            
                
        return self.A[absorber] * self.fheat * HeatingRate
            
    def SecondaryIonizationRate(self, absorber, donor):
        """
        Secondary ionization rate which we denote elsewhere as gamma (note little g).
        
            species = species being ionized by photo-electron
            donor_species = species the photo-electron came from
            
        If this routine is called, it means TabulateIntegrals = 1.    
        """    
        
        if self.esec.Method < 2:
            IonizationRate = self.PsiN[donor].copy()
            IonizationRate -= self.grid.ioniz_thresholds[donor] \
                * erg_per_ev * self.PhiN[donor]
            if self.pf['photon_conserving']:
                IonizationRate -= self.PsiNdN[donor]
                IonizationRate += self.grid.ioniz_thresholds[donor] \
                    * erg_per_ev * self.PhiNdN[donor]
                            
        else:
            
            IonizationRate = self.PsiWiggleN[absorber][donor] \
                - self.grid.ioniz_thresholds[donor] \
                * erg_per_ev * self.PhiWiggleN[absorber][donor]
            if self.pf['photon_conserving']:
                IonizationRate -= self.PsiWiggleNdN[absorber][donor]
                IonizationRate += self.grid.ioniz_thresholds[donor] \
                    * erg_per_ev * self.PhiWiggleNdN[absorber][donor]
                        
        # Normalization (by number densities) will be applied in 
        # chemistry solver    
            
        return self.A[donor] * self.fion[absorber] * IonizationRate \
                / self.grid.ioniz_thresholds[absorber] / erg_per_ev    
        
        
        