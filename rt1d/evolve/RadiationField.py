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
        self.esec = SecondaryElectrons(**kwargs)
    
    def ColumnDensity(self, data):
        """
        Compute column densities for all absorbing species.
        """    
        
        N = {}
        for absorber in self.grid.absorbers:
            N[absorber] = np.cumsum(data[absorber] \
                * self.grid.x_to_n[absorber] * self.grid.dr)
                    
        return N            
                    
    #def OpticalDepth(self, data, N = None):
    #    """
    #    Compute optical depths for all absorbing species.
    #    """
    #    
    #    if N is None:
    #        N = self.ColumnDensity(data)
    #        
    #    tau = {}
    #    for absorber in self.grid.absorbers:
    #        sigma = BoundFreeAbsorptionCrossSection(absorber)
    #        tau[absorber] = N[absorber] * sigma
    #    
    
    def SourceDependentCoefficients(self, data, t):
        """
        Compute rate coefficients for photo-ionization, secondary ionization, 
        and photo-heating.
        """

        k_H = np.zeros_like(self.grid.zeros_grid_x_absorbers)
        Gamma = np.zeros_like(self.grid.zeros_grid_x_absorbers)
        gamma = np.zeros_like(self.grid.zeros_grid_x_absorbers2)
                
        if not self.src.SourceOn(t):
            return Gamma, gamma, k_H
            
        N = self.ColumnDensity(data)    
        
        # Loop over absorbing species
        for i, absorber in enumerate(self.grid.absorbers):
            
            n = data[absorber] * self.grid.x_to_n[absorber] 
                
            """
            Photo-ionization for discrete spectrum (multi-freq. approach).
            """    
            
            if self.src.multi_freq:
                            
                # Loop over energy groups
                Gamma_E = np.zeros([self.grid.dims, self.src.Nfreq])
                for j, E in enumerate(self.src.E):
                    
                    if E < self.grid.ioniz_thresholds[i]:
                        continue    
                                        
                    # Optical depth up to cells at energy E
                    tau_r = N[absorber] * self.src.sigma[j]
                    
                    # Optical depth of cells (at this photon energy)                                                           
                    tau_c = self.grid.dr * n * self.src.sigma[j]
                                                                    
                    # Photo-ionization by *this* energy group
                    Gamma_E[...,j] = \
                        self.PhotoIonizationRateMultiFreq(self.src.Qdot[j], n, 
                        tau_r, tau_c)
                     
                    # Heating    
                    if self.grid.isothermal:
                        continue
                         
                    fheat = self.esec.DepositionFraction(E = E,
                        xHII = data['h_2'], channel = 'heat')
                    
                    # Total energy deposition rate per atom i via photo-electrons 
                    # due to ionizations by *this* energy group. 
                    ee = Gamma_E[...,j] * (E - self.grid.ioniz_thresholds[i]) * erg_per_ev 
                    
                    k_H[...,i] += ee * fheat
                        
                    if not self.pf['SecondaryIonization']:
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
                        gamma[...,k,i] += ee * fion \
                            / (self.grid.ioniz_thresholds[k] * erg_per_ev)
                                                                                   
                # Total photo-ionization tally
                Gamma[...,i] = np.sum(Gamma_E, axis = 1)
                
            elif self.src.multi_grp:
                pass
            
            else:
                pass                
                
        return Gamma, gamma, k_H
        
    def PhotoIonizationRateMultiFreq(self, qdot, n, tau_r, tau_c):
        """
        Returns photo-ionization rate coefficient for single frequency over
        the entire grid.
        """     
                                        
        q0 = qdot * np.exp(-tau_r)               # number of photons entering cell per sec
        dq = q0 * (1. - np.exp(-tau_c))          # number of photons absorbed in cell per sec
        IonizationRate = dq / n / self.grid.Vsh  # ionizations / sec / atom        
                          
        return IonizationRate
        
    def PhotoIonizationRateMultiGrp(self):
        pass    
        
    def PhotoIonizationRate(self, n, qdot, tau_r, tau_c):
        """
        Returns photo-ionization rate coefficient which we denote elsewhere as Gamma.  
        
            [IonizationRate] = 1 / s
            
            Inputs:
            E = Energy of photons in ray (eV)
            Qdot = Photon luminosity of source for this ray (s^-1)
            Lbol = Bolometric luminosity of source (erg/s)
                        
        """     
                    
        Phi_N = Phi_N_dN = None
                                        
        if rs.pf['ForceIntegralTabulation'] or not rs.pf['DiscreteSpectrum']:
            if rs.pf['PhotonConserving']:
                Phi_N = rs.Interpolate.interp(indices_in, "logPhi%i" % species, 
                    [ncol[0], ncol[1], ncol[2], x_HII, t])
                Phi_N_dN = rs.Interpolate.interp(indices_out, "logPhi%i" % species, 
                    [nout[0], nout[1], nout[2], x_HII, t])
                IonizationRate = Phi_N - Phi_N_dN
            else:                
                Phi_N = rs.Interpolate.interp(indices_in, "logPhi%i" % species, 
                    [ncol[0], ncol[1], ncol[2], x_HII, t])       
                IonizationRate = Phi_N
        