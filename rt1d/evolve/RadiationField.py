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
        self.sigma = (np.ones([self.grid.dims, self.src.Nfreq]) * self.src.sigma).T                  
                      
    def SourceDependentCoefficients(self, data, t):
        """
        Compute rate coefficients for photo-ionization, secondary ionization, 
        and photo-heating.
        """

        self.k_H = np.zeros_like(self.grid.zeros_grid_x_absorbers)
        self.Gamma = np.zeros_like(self.grid.zeros_grid_x_absorbers)
        self.gamma = np.zeros_like(self.grid.zeros_grid_x_absorbers2)
                
        if not self.src.SourceOn(t):
            return Gamma, gamma, k_H
            
        self.N = self.grid.ColumnDensity(data)
        
        # Eventually loop over sources here
        
        # Loop over absorbing species
        for absorber in self.grid.absorbers:
                            
            # Discrete spectrum (multi-freq approach)
            if self.src.multi_freq:
                self.MultiFreqCoefficients(data, absorber)   
            
            # Discrete spectrum (multi-grp approach)
            elif self.src.multi_group:
                pass
            
            # Continuous spectrum (uses tabulated integral values)
            else:
                self.TabulatedCoefficients(data, absorber)
                
        return self.Gamma, self.gamma, self.k_H
        
    def MultiFreqCoefficients(self, data, absorber):
        """
        Compute all source-dependent rates for given absorber assuming a
        multi-frequency SED.
        """   
        
        i = self.grid.absorbers.index(absorber)
        n = data[absorber] * self.grid.x_to_n[absorber] 
        N = self.N[absorber]
               
        # Optical depth up to cells at energy E
        N = np.ones([self.src.Nfreq, self.grid.dims]) * self.N[absorber]
        
        self.tau_r = N * self.sigma
        self.tau_tot = np.sum(self.tau_r, axis = 1)
                
        # Loop over energy groups
        self.Gamma_E = np.zeros([self.grid.dims, self.src.Nfreq])
        for j, E in enumerate(self.src.E):
            
            if E < self.grid.ioniz_thresholds[i]:
                continue    
            
            # Optical depth of cells (at this photon energy)                                                           
            tau_c = self.grid.dr * n * self.src.sigma[j]
                                                            
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
            ee = self.Gamma_E[...,j] * (E - self.grid.ioniz_thresholds[i]) \
               * erg_per_ev 
            
            self.k_H[...,i] += ee * fheat
                
            if not self.pf['secondary_ionization']:
                continue
                                        
            # Ionizations of species k by photoelectrons from species i
            # Neglect HeII until somebody figures out how that works
            for k, otherabsorber in enumerate(self.grid.absorbers):
            
                # If these photo-electrons don't have enough 
                # energy to ionize species k, continue    
                if (E - self.grid.ioniz_thresholds[i]) < \
                    self.grid.ioniz_thresholds[k]:
                    continue    
                
                fion = self.esec.DepositionFraction(E = E, 
                    xHII = data['h_2'], channel = absorber)
                                                                        
                # (This k) = i from paper, and (this i) = j from paper
                self.gamma[...,k,i] += ee * fion \
                    / (self.grid.ioniz_thresholds[k] * erg_per_ev)
                                                                           
        # Total photo-ionization tally
        self.Gamma[...,i] = np.sum(self.Gamma_E, axis = 1)
    
    def TabulatedCoefficients(self, data, absorber):
        pass
                
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
        
    def PhotoIonizationRate(self, n, qdot, tau_r, tau_c):
        """
        Returns photo-ionization rate coefficient for continuous source.
        """     
                    
        Phi_N = Phi_N_dN = None
                                        
        if rs.pf['photon_conserving']:
            Phi_N = rs.Interpolate.interp(indices_in, "logPhi%i" % species, 
                [ncol[0], ncol[1], ncol[2], x_HII, t])
            Phi_N_dN = rs.Interpolate.interp(indices_out, "logPhi%i" % species, 
                [nout[0], nout[1], nout[2], x_HII, t])
            IonizationRate = Phi_N - Phi_N_dN
        else:                
            Phi_N = rs.Interpolate.interp(indices_in, "logPhi%i" % species, 
                [ncol[0], ncol[1], ncol[2], x_HII, t])       
            IonizationRate = Phi_N
        