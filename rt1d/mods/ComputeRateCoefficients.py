"""

ComputeRateCoefficients.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Tue Jan 17 16:57:48 2012

Description: 

"""

import numpy as np
from . import ComputeCrossSections 
from ComputeCrossSections import PhotoIonizationCrossSection
from . import SecondaryElectrons
from SecondaryElectrons import SecondaryElectrons

class RateCoefficients:
    def __init__(self, pf, itabs = None):
        self.pf = pf
        self.itabs = itabs
        self.esec = SecondaryElectrons(pf)
        
        self.PhotonConserving = pf["PhotonConserving"]        
        self.CollisionalIonization = pf["CollisionalIonization"]
        
    def PhotoIonization(self, species = None, E = None, Qdot = None, Lbol = None, 
        ncol = None, n_e = None, n_HI = None, n_HeI = None, x_HII = None, 
        T = None, r = None, dr = None, dt = None, indices = None):
        """
        Returns photo-ionization rate coefficient which we denote elsewhere as Gamma.  
        
            returns IonizationRate: [IonizationRate] = 1 / s
            
            Inputs:
            E = Energy of photons in ray (eV)
            Qdot = Photon luminosity of source for this ray (s^-1)
            Lbol = Bolometric luminosity of source (erg/s)
            
        """     
        
        if self.PhotonConserving:
            sigma = PhotoIonizationCrossSection(E, species = species)
            
            if sigma == 0:
                IonizationRate = 0
            else:
                tau = ncol[species] * sigma
                tau_c = dr * n_HI * sigma
                Vcell = dr 
                Q0 = Qdot * np.exp(-tau) / 4. / np.pi / r**2       # number of photons entering cell
                dQ = Q0 * (1. - np.exp(-tau_c))
                                    
                IonizationRate = dQ / dt / n_HI / Vcell    
          
        else:   
            # Photo-Ionization       
            IonizationRate = Lbol * \
                             self.Interpolate.interp(indices, "PhotoIonizationRate0", ncol)
                                     
            if self.SecondaryIonization:
                IonizationRate += Lbol * \
                                  self.esec.DepositionFraction(0.0, x_HII, channel = 1) * \
                                  self.Interpolate.interp(indices, "SecondaryIonizationRateHI0", ncol)    
                                                          
                if self.MultiSpecies > 0:
                    IonizationRate += Lbol * (n_HeI / n_HI) * \
                                     self.esec.DepositionFraction(0.0, x_HII, channel = 1) * \
                                     self.Interpolate.interp(indices, "SecondaryIonizationRateHI1", ncol)
            
            if not self.PlaneParallelField: IonizationRate /= (4. * np.pi * r**2)
                                                    
            # Collisional Ionization
            if self.CollisionalIonization:
                IonizationRate += n_e * 5.85e-11 * np.sqrt(T) * (1. + np.sqrt(T / 1.e5))**-1. * np.exp(-1.578e5 / T)
                    
        return IonizationRate
        
    def SecondaryIonization(self, species = None, E = None, Qdot = None, Lbol = None, 
        ncol = None, n_e = None, n_HI = None, n_HeI = None, x_HII = None, 
        T = None, r = None, dr = None, dt = None, indices = None):
        """
        Secondary ionization rate which we denote elsewhere as gamma (note little g).
        """    
        
        IonizationRate += Lbol * \
            self.esec.DepositionFraction(0.0, x_HII, channel = 1) * \
            self.Interpolate.interp(indices, "SecondaryIonizationRateHI0", ncol)    
                                                          
        if self.MultiSpecies > 0:
            IonizationRate += Lbol * (n_HeI / n_HI) * \
                self.esec.DepositionFraction(0.0, x_HII, channel = 1) * \
                self.Interpolate.interp(indices, "SecondaryIonizationRateHI1", ncol)
        
        return IonizationRate
        
    def CollisionalIonization(self, species = None, n_e = None, T = None):
        """
        Secondary ionization rate which we denote elsewhere as Beta (note little g).
        """    
        
        if not self.CollisionalIonization:
            return 0.0
           
        if species == 0:  
            return n_e * 5.85e-11 * np.sqrt(T) * (1. + np.sqrt(T / 1.e5))**-1. * np.exp(-1.578e5 / T)    
          
        if species == 1:    
            return 2.38e-11 * np.sqrt(T) * (1. + np.sqrt(T / 1.e5))**-1. * np.exp(-2.853e5 / T) 
        
        if species == 2:
            return 5.68e-12 * np.sqrt(T) * (1. + np.sqrt(T / 1.e5))**-1. * np.exp(-6.315e5 / T) 
            
    def RadiativeRecombination(self, T = None):
        """
        Coefficient for radiative recombination of hydrogen.
        """
        return 2.6e-13 * (T / 1.e4)**-0.85        
        
    def DielectricRecombination(self, T = None):
        """
        Dielectric recombination coefficient for Helium.
        """
        return 1.9e-3 * T**-1.5 * np.exp(-4.7e5 / T) * (1. + 0.3 * np.exp(-9.4e4 / T))    
                    
    def IonizationRateCoefficientHeI(self, ncol, n_HI, n_HeI, x_HII, T, r, Lbol, indices):
        """
        Returns ionization rate coefficient for HeI, which we denote elsewhere as Gamma_HeI.  Includes photo 
        and secondary ionizations from fast electrons.  Unlike the hydrogen case, the collisional ionizations
        are included in the rate equation itself instead of a coefficient.
        
            units: 1 / s
        """                
        
        IonizationRate = Lbol * \
                         self.Interpolate.interp(indices, "PhotoIonizationRate1", ncol)
        
        if self.SecondaryIonization:
            IonizationRate += Lbol * \
                              self.esec.DepositionFraction(0.0, x_HII, channel = 2) * \
                              self.Interpolate.interp(indices, "SecondaryIonizationRateHeI1", ncol)
            
            IonizationRate += (n_HI / n_HeI) * Lbol * \
                              self.esec.DepositionFraction(0.0, x_HII, channel = 2) * \
                              self.Interpolate.interp(indices, "SecondaryIonizationRateHeI1", ncol) 
        
        if not self.PlaneParallelField: 
            IonizationRate /= (4. * np.pi * r**2)
        
        return IonizationRate
        
    def IonizationRateCoefficientHeII(self, ncol, n_HI, n_HeI, x_HII, T, r, Lbol, indices):
        """
        Returns ionization rate coefficient for HeII, which we denote elsewhere as Gamma_HeII.  Includes photo 
        and secondary ionizations from fast electrons.  Unlike the hydrogen case, the collisional ionizations
        are included in the rate equation itself instead of a coefficient.  Note: TZ07 do not include secondary
        helium II ionizations, but I am going to.
        
            units: 1 / s
        """       
        
        IonizationRate = Lbol * \
                         self.Interpolate.interp(indices, "PhotoIonizationRate2", ncol) \
        
        if self.SecondaryIonization > 1:
            IonizationRate += Lbol * self.esec.DepositionFraction(0.0, x_HII, channel = 3) * \
                self.Interpolate.interp(indices, "SecondaryIonizationRate2", ncol)
                        
        if not self.PlaneParallelField: 
            IonizationRate /= (4. * np.pi * r**2)
        
        return IonizationRate
        
    def HeatGain(self, ncol, nabs, x_HII, r, Lbol, indices):
        """
        Returns the total heating rate at radius r and time t.  These are all the terms in Eq. 12 of TZ07 on
        the RHS that are positive.
        
            units: erg / s / cm^3
        """
                                 
        heat = nabs[0] * self.Interpolate.interp(indices, "ElectronHeatingRate0", ncol)
    
        if self.MultiSpecies > 0:
            heat += nabs[1] * self.Interpolate.interp(indices, "ElectronHeatingRate1", ncol)
            heat += nabs[2] * self.Interpolate.interp(indices, "ElectronHeatingRate2", ncol)
                                                           
        heat *= Lbol * self.esec.DepositionFraction(0.0, x_HII, channel = 0)
        
        if not self.PlaneParallelField: heat /= (4. * np.pi * r**2)
                                                                                                                                                                                                                         
        return heat
    
    def HeatLoss(self, nabs, nion, n_e, n_B, T, z, mu):
        """
        Returns the total cooling rate for a cell of temperature T and with species densities given in 'nabs', 'nion', and 'n_e'. 
        This quantity is the sum of all terms on the RHS of Eq. 12 in TZ07 that are negative, 
        though we do not apply the minus sign until later, in 'ThermalRateEquation'.
        
            units: erg / s / cm^3
        """
            
        T_cmb = 2.725 * (1. + z)
        cool = 0.
        
        # Cooling by collisional ionization
        for i, n in enumerate(nabs):
            cool += n * self.CollisionalIonizationCoolingCoefficient(T, i)
                
        # Cooling by recombinations
        for i, n in enumerate(nion):
            cool += n * self.RecombinationCoolingCoefficient(T, i)
            
        # Cooling by dielectronic recombination
        cool += nion[2] * self.DielectricRecombinationCoolingCoefficient(T)
                
        # Cooling by collisional excitation
        if self.CollisionalExcitation:
            for i, n in enumerate(nabs):
                cool += n * self.CollisionalExcitationCoolingCoefficient(T, nabs, nion, i)
        
        # Compton cooling - from FK96
        if self.ComptonCooling:
            cool += 4. * k_B * (T - T_cmb) * (np.pi**2 / 15.) * (k_B * T_cmb / hbar / c)**3 * (k_B * T_cmb / m_e / c**2) * sigma_T * c
        
        # Cooling by free-free emission
        if self.FreeFreeEmission:
            cool += (nion[0] + nion[1] + 4. * nion[2]) * 1.42e-27 * 1.1 * np.sqrt(T) # Check on Gaunt factor        
                
        cool *= n_e
        
        # Hubble cooling
        if self.CosmologicalExpansion:
            cool += 2. * self.cosmo.HubbleParameter(z) * (k_B * T * n_B / mu)
                                
        return cool
        
    def CollisionalIonizationCoolingCoefficient(self, T, species):
        """
        Returns coefficient for cooling by collisional ionization.  These are equations B4.1a, b, and d respectively
        from FK96.
        
            units: erg cm^3 / s
        """
        
        if species == 0: return 1.27e-21 * np.sqrt(T) * (1. + np.sqrt(T / 1e5))**-1. * np.exp(-1.58e5 / T)
        if species == 1: return 9.38e-22 * np.sqrt(T) * (1. + np.sqrt(T / 1e5))**-1. * np.exp(-2.85e5 / T)
        if species == 2: return 4.95e-22 * np.sqrt(T) * (1. + np.sqrt(T / 1e5))**-1. * np.exp(-6.31e5 / T)
    
    def CollisionalExcitationCoolingCoefficient(self, T, nabs, nion, species):
        """
        Returns coefficient for cooling by collisional excitation.  These are equations B4.3a, b, and c respectively
        from FK96.
        
            units: erg cm^3 / s
        """
        
        if species == 0: return 7.5e-19 * (1. + np.sqrt(T / 1e5))**-1. * np.exp(-1.18e5 / T)
        if species == 1: 
            if self.MultiSpecies == 0: return 0.0
            else: return 9.1e-27 * T**-0.1687 * (1. + np.sqrt(T / 1e5))**-1. * np.exp(-1.31e4 / T) * nion[1] / nabs[1]   # CONFUSION
        if species == 2: 
            if self.MultiSpecies == 0: return 0.0
            else: return 5.54e-17 * T**-0.397 * (1. + np.sqrt(T / 1e5))**-1. * np.exp(-4.73e5 / T)    
        
    def RecombinationCoolingCoefficient(self, T, species):
        """
        Returns coefficient for cooling by recombination.  These are equations B4.2a, b, and d respectively
        from FK96.
        
            units: erg cm^3 / s
        """
        
        if species == 0: return 6.5e-27 * T**0.5 * (T / 1e3)**-0.2 * (1.0 + (T / 1e6)**0.7)**-1.0
        if species == 1: return 1.55e-26 * T**0.3647
        if species == 2: return 3.48e-26 * np.sqrt(T) * (T / 1e3)**-0.2 * (1. + (T / 4e6)**0.7)**-1.
        
    def DielectricRecombinationCoolingCoefficient(self, T):
        """
        Returns coefficient for cooling by dielectric recombination.  This is equation B4.2c from FK96.
        
            units: erg cm^3 / s
        """
        return 1.24e-13 * T**-1.5 * np.exp(-4.7e5 / T) * (1. + 0.3 * np.exp(-9.4e4 / T))
    
    def OpticalDepth(self, n):
        """
        Returns the total optical depth over all energies due to column densities of HI, HeI, and HeII, which
        are stored in the variable 'n' as a three element array.
        """
        
        func = lambda E: PhotoIonizationCrossSection(E, 0) * n[0] + PhotoIonizationCrossSection(E, 1) * n[1] \
            + PhotoIonizationCrossSection(E, 2) * n[2]
            
        return quad(func, self.rs.Emin, self.rs.Emax, epsrel = 1e-8)[0]   