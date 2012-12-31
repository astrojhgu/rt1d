"""

RadiationField.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Dec 31 11:16:59 2012

Description: 

"""

import numpy as np
from ..util import parse_kwargs

#E_th = {'h_1': 13.6, 'he_1': 24.6, 'he_2': 54.4}

class RadiationField:
    def __init__(self, grid, source, **kwargs):
        self.pf = parse_kwargs(**kwargs)
        self.grid = grid
        self.src = source
        
        self.x_to_n = {}
        for absorber in self.grid.absorbers:
            self.x_to_n[absorber] = self.grid.n_H \
                * self.grid.species_abundances[absorber]
    
    def ColumnDensity(self, data):
        """
        Compute column densities for all absorbing species.
        """    
        
        N = {}
        for absorber in self.grid.absorbers:
            N[absorber] = np.cumsum(data[absorber] \
                * self.x_to_n[absorber] * self.grid.dr)
                    
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
    
    def SourceDependentCoefficients(self, data):
        """
        Compute rate coefficients for photo-ionization, secondary ionization, 
        and photo-heating.
        """

        N = self.ColumnDensity(data)

        Gamma = np.zeros_like(self.grid.zeros_grid_x_absorbers)
        for i, absorber in enumerate(self.grid.absorbers):
            Gamma[...,i] = np.zeros(self.grid.dims)
            
            n = data[absorber] * self.x_to_n[absorber]
                        
            if not self.src.discrete: 
                continue   
                
            """
            Photo-ionization for discrete spectrum (multi-freq. approach).
            """    
                            
            # Loop over energy groups
            Gamma_E = np.zeros([self.grid.dims, self.src.Nfreq])
            for j, E in enumerate(self.src.E):
                
                #if E < E_th[i]:
                #    continue    
                
                #A = self.src.Qdot[j] / n / self.grid.Vsh
                    
                # Optical depth up to cells at energy E
                tau_r = N[absorber] * self.src.sigma[j]
                
                # Optical depth of cells (at this photon energy)                                                           
                tau_c = self.grid.dr * n * self.src.sigma[j]
                                                                
                # Photo-ionization by *this* energy group
                Gamma_E[...,j] = \
                    self.PhotoIonizationRateMultiFreq(self.src.Qdot[j], n, 
                    tau_r, tau_c)
                                                                
                # Total photo-ionization tally
                Gamma[...,i] += Gamma_E[...,j]
                
                #fheat = self.esec.DepositionFraction(E = E, xHII = x_HII, channel = 0)
                
                # Total energy deposition rate per atom i via photo-electrons 
                # due to ionizations by *this* energy group. 
                #ee = Gamma_E[j] * (E - E_th[i]) * erg_per_ev 
                #    
                #if self.pf['SecondaryIonization']:
                #    
                #    # Ionizations of species k by photoelectrons from species i
                #    # Neglect HeII until somebody figures out how that works
                #    for k in xrange(2):
                #        if not self.pf['MultiSpecies'] and k > 0:
                #            continue
                #        
                #        # If these photo-electrons dont have enough energy to ionize species k, continue    
                #        if (E - E_th[i]) < E_th[k]:
                #            continue    
                #        
                #        fion = self.esec.DepositionFraction(E = E, xHII = x_HII, channel = k + 1)
                #        
                #        # (This k) = i from paper, and (this i) = j from paper
                #        gamma[k][i] += ee * fion / (E_th[k] * erg_per_ev)
                #                                                                                      
                #if self.pf['Isothermal']:
                #    continue                           
                #                                    
                ## Heating rate coefficient        
                #k_H[i] += ee * fheat
        
        return Gamma    
        
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
        