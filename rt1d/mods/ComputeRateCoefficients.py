"""

ComputeRateCoefficients.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Tue Jan 17 16:57:48 2012

Description: 

"""

import numpy as np
from . import ComputeCrossSections, SecondaryElectrons, Interpolate
from ComputeCrossSections import PhotoIonizationCrossSection
from SecondaryElectrons import SecondaryElectrons
from Interpolate import Interpolate

E_th = [13.6, 24.6, 54.4]
erg_per_ev = 1.60217646e-19 / 1e-7 

class RateCoefficients:
    def __init__(self, pf, rs, itabs = None, n_col = None):
        self.pf = pf        
        self.rs = rs
        self.esec = SecondaryElectrons(pf)
        
        # Initialize integral tables
        if type(itabs) is list:
            self.itabs = itabs[0]
            self.itabs_fback = itabs[1]
            self.Interpolate = Interpolate(pf, n_col, self.itabs)
            self.Interpolate_fback = Interpolate(pf, n_col, self.itabs_fback)
        else:
            self.itabs = itabs
            self.itabs_fback = self.Interpolate_fback = None
            self.Interpolate = Interpolate(pf, n_col, self.itabs)
            
        self.TabulateIntegrals = self.pf["TabulateIntegrals"]
        self.AutoFallback = self.pf['AutoFallback']
        self.Fallback = self.pf["PhotonConserving"] and self.pf['AutoFallback']
        
        # Physics parameters
        self.MultiSpecies = pf["MultiSpecies"]
        self.Isothermal = pf["Isothermal"]
        self.PhotonConserving = pf["PhotonConserving"]        
        self.CollisionalIonization = pf["CollisionalIonization"]
        self.SecondaryIonization = pf["SecondaryIonization"]
        self.PlaneParallelField = pf["PlaneParallelField"]
        
    def ConstructArgs(self, args, indices, Lbol, r, ncol, T, dx, tau):
        """
        Make list of rate coefficients that we'll pass to solver.
        
            args = (nabs, n_H, n_He, n_e, Gamma, gamma, Beta, alpha, k_H, zeta, eta, psi)
        """    
        
        nabs = args[0]
        nion = args[1]
        n_H = args[1]
        x_HII = (n_H - nabs[0]) / n_H
        n_He = args[2]
        n_e = args[3]
        
        Gamma = np.zeros(3)
        gamma = np.zeros(3)
        alpha = np.zeros(3)
        Beta = np.zeros(3)
        k_H = np.zeros(3)
        zeta = np.zeros(3)
        eta = np.zeros(3)
        psi = np.zeros(3)
        xi = np.zeros(3)
        
        # Standard - integral tabulation
        if self.TabulateIntegrals:
            
            # Loop over species   
            for i in xrange(3):
                
                if not self.MultiSpecies and i > 0:
                    continue
                
                # Ionization
                Gamma[i] = self.PhotoIonizationRate(species = i, indices = indices,  
                    Lbol = Lbol, r = r, ncol = ncol, nabs = nabs, tau = tau, dr = dx)
               
                gamma[i] = self.SecondaryIonizationRate(indices = indices)
                Beta[i] = self.CollisionalIonizationRate(species = i, T = T, n_e = n_e)
                alpha[i] = self.RadiativeRecombinationRate(species = i, T = T)
                
                # Dielectric recombination
                if i == 2:
                    xi[i] = self.DielectricRecombinationRate(T = T)
                
                if self.Isothermal:
                    continue
                
                # Heating/cooling            
                k_H[i] = self.PhotoElectricHeatingRate(species = i, Lbol = Lbol, r = r, 
                    x_HII = x_HII, indices = indices, ncol = ncol, dr = dx, nabs = nabs)
                    
                zeta[i] = self.CollisionalIonizationCoolingRate(species = i, T = T)    
                eta[i] = self.RecombinationCoolingRate(species = i, T = T) 
                psi[i] = self.CollisionalExcitationCoolingRate(species = i, T = T, nabs = nabs, nion = nion)
        
        # Only the photon-conserving algorithm is capable of this                                                  
        else:
               
            # Loop over species   
            for i in xrange(3):
                
                if not self.MultiSpecies and i > 0:
                    continue
                
                Gamma_E = np.zeros_like(self.rs.E)
                for j, E in enumerate(self.rs.E):
                
                    Gamma_E[j] = self.PhotoIonizationRate(E = E, Qdot = self.rs.Qdot[j], ncol = ncol,
                        nabs = nabs, r = r, dr = dx, species = i, tau = tau, Lbol = Lbol)
                        
                    # Ionization
                    Gamma[i] += Gamma_E[j]
                    
                    if not self.Isothermal:
                        ee = Gamma_E[j] * (E - E_th[i]) # Total photo-electron energy
                        
                        # Photo-heating           
                        k_H[i] += ee * self.esec.DepositionFraction(E = E, xi = x_HII, channel = 0) 
                        
                    if self.SecondaryIonization:
                        gamma[i] += ee * self.esec.DepositionFraction(E = E, xi = x_HII, channel = 1 + i)
                    
                # Collisional ionization + radiative recombination + cooling (no dep. on radiation field)    
                alpha[i] = self.RadiativeRecombinationRate(species = i, T = T)
                Beta[i] = self.CollisionalIonizationRate(species = i, T = T, n_e = n_e)
                
                # Dielectric recombination
                if i == 2:
                    xi[i] = self.DielectricRecombinationRate(T = T)
                
                if self.Isothermal:
                    continue
                
                zeta[i] += self.CollisionalIonizationCoolingRate(species = i, T = T)  
                eta[i] = self.RecombinationCoolingRate(species = i, T = T) 
                psi[i] = self.CollisionalExcitationCoolingRate(species = i, T = T, nabs = nabs, nion = nion)               

            k_H *= erg_per_ev
            gamma *= erg_per_ev

        return [Gamma, gamma, Beta, alpha, k_H, zeta, eta, psi, xi]
        
    def ShellVolume(self, r, dr):
        """
        Return volume of shell at distance r, thickness dr.
        """
        return 4. * np.pi * ((r + dr)**3 - r**3) / 3.

    def PhotoIonizationRate(self, species = None, E = None, Qdot = None, Lbol = None, 
        ncol = None, n_e = None, nabs = None, x_HII = None, 
        T = None, r = None, dr = None, tau = None, indices = None):
        """
        Returns photo-ionization rate coefficient which we denote elsewhere as Gamma.  
        
            returns IonizationRate: [IonizationRate] = 1 / s
            
            Inputs:
            E = Energy of photons in ray (eV)
            Qdot = Photon luminosity of source for this ray (s^-1)
            Lbol = Bolometric luminosity of source (erg/s)
            
        """     
           
        if self.TabulateIntegrals:
            if self.PhotonConserving:
                nout = ncol + dr * nabs    # Column density up to and *including* this cell
                ncell = nout - ncol
                if ncell[species] < self.Interpolate.MinimumColumns[species] and self.Fallback:
                    A = Lbol / 4. / np.pi / r**2
                    IonizationRate = self.Interpolate_fback.interp(indices, "PhotoIonizationRate%i" % species, ncol)
                else:
                    A = Lbol / nabs[species] / self.ShellVolume(r, dr)
                    incident = self.Interpolate.interp(indices, "PhotoIonizationRate%i" % species, ncol)
                    outgoing = self.Interpolate.interp(indices, "PhotoIonizationRate%i" % species, nout)
                    IonizationRate = incident - outgoing  
                
            else:
                A = Lbol / 4. / np.pi / r**2
                IonizationRate = self.Interpolate.interp(indices, "PhotoIonizationRate%i" % species, ncol)       
           
        else:
            sigma = PhotoIonizationCrossSection(E, species = species)
            tau_E = ncol[species] * sigma       # Optical depth up until this cell
            tau_c = dr * nabs[species] * sigma  # Optical depth of this cell
            Vcell = 4. * np.pi * ((r + dr)**3 - r**3) / 3.
            Q0 = Qdot * np.exp(-tau_E)          # number of photons entering cell per sec
            dQ = Q0 * (1. - np.exp(-tau_c))     # number of photons absorbed in cell per sec
            
            A = 1.          
            IonizationRate = dQ / nabs[species] / Vcell    # ionizations / sec / hydrogen atom
                                                                                                                        
        return A * IonizationRate
        
    def CollisionalIonizationRate(self, species = None, n_e = None, T = None):
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
        
    def SecondaryIonizationRate(self, species1 = None, species2 = None, E = None, Qdot = None, Lbol = None, 
        ncol = None, n_e = None, n_HI = None, n_HeI = None, x_HII = None, 
        T = None, r = None, dr = None, dt = None, indices = None):
        """
        Secondary ionization rate which we denote elsewhere as gamma (note little g).
        
            species1 = species experiencing primary photo-ionization
            species2 = species being ionized by photo-electron
        
        """    
        
        if not self.SecondaryIonization:
            return 0.0
        
        if self.PhotonConserving:
            pass
        
        else:
            IonizationRate = Lbol * self.Interpolate.interp(indices, "SecondaryIonizationRateHI0", ncol)                
        
        IonizationRate += Lbol * \
            self.esec.DepositionFraction(0.0, x_HII, channel = 1) * \
            self.Interpolate.interp(indices, "SecondaryIonizationRateHI0", ncol)    
                                                          
        if self.MultiSpecies > 0:
            IonizationRate += Lbol * (n_HeI / n_HI) * \
                self.esec.DepositionFraction(0.0, x_HII, channel = 1) * \
                self.Interpolate.interp(indices, "SecondaryIonizationRateHI1", ncol)
        
        return IonizationRate 
        
    def RadiativeRecombinationRate(self, species = 0, T = None):
        """
        Coefficient for radiative recombination.  Here, species = 0, 1, 2
        refers to HII, HeII, and HeIII.
        """
        
        if species == 0:
            return 2.6e-13 * (T / 1.e4)**-0.85 
        elif species == 1:
            return 9.94e-11 * T**-0.6687
        else:
            alpha = 3.36e-10 * T**-0.5 * (T / 1e3)**-0.2 * (1. + (T / 4.e6)**0.7)**-1
            if T < 2.2e4: 
                alpha *= (1.11 - 0.044 * np.log(T)) # To n >= 1                       
            else: 
                alpha *= (1.43 - 0.076 * np.log(T)) # To n >= 2
            
            return alpha        
        
    def DielectricRecombinationRate(self, T = None):
        """
        Dielectric recombination coefficient for Helium.
        """
        return 1.9e-3 * T**-1.5 * np.exp(-4.7e5 / T) * (1. + 0.3 * np.exp(-9.4e4 / T)) 
        
    def PhotoElectricHeatingRate(self, species = None, E = None, Qdot = None, Lbol = None, 
        ncol = None, n_e = None, nabs = None, x_HII = None, 
        T = None, r = None, dr = None, dt = None, indices = None):
        """
        Photo-electric heating rate coefficient due to photo-electrons previously 
        bound to `species.'  If this method is called, it means TabulateIntegrals = 1.
        """
        
        if self.PhotonConserving:
            nout = ncol + dr * nabs    # Column density up to and *including* this cell
            ncell = nout - ncol
            if ncell[species] < self.Interpolate.MinimumColumns[species] and self.Fallback:
                A = Lbol / 4. / np.pi / r**2            
                heat = self.Interpolate_fback.interp(indices, "ElectronHeatingRate%i" % species, ncol)
            else:
                A = Lbol / nabs[species] / self.ShellVolume(r, dr)  
                incident = self.Interpolate.interp(indices, "ElectronHeatingRate%i" % species, ncol)
                outgoing = self.Interpolate.interp(indices, "ElectronHeatingRate%i" % species, nout)
                heat = incident - outgoing  
                    
                
        else:
            A = Lbol / 4. / np.pi / r**2            
            heat = self.Interpolate.interp(indices, "ElectronHeatingRate%i" % species, ncol)

        return A * self.esec.DepositionFraction(0.0, x_HII, channel = 0) * heat
        
    def CollisionalIonizationCoolingRate(self, species = 0, T = None):
        """
        Returns coefficient for cooling by collisional ionization.  These are equations B4.1a, b, and d respectively
        from FK96.
        
            units: erg cm^3 / s
        """
        
        if self.Isothermal:
            return 0
        
        if species == 0: 
            return 1.27e-21 * np.sqrt(T) * (1. + np.sqrt(T / 1e5))**-1. * np.exp(-1.58e5 / T)
        if species == 1: 
            return 9.38e-22 * np.sqrt(T) * (1. + np.sqrt(T / 1e5))**-1. * np.exp(-2.85e5 / T)
        if species == 2: 
            return 4.95e-22 * np.sqrt(T) * (1. + np.sqrt(T / 1e5))**-1. * np.exp(-6.31e5 / T)
            
    def CollisionalExcitationCoolingRate(self, species = 0, T = None, nabs = None, nion = None):
        """
        Returns coefficient for cooling by collisional excitation.  These are equations B4.3a, b, and c respectively
        from FK96.
        
            units: erg cm^3 / s
        """
        
        if self.Isothermal:
            return 0
        
        if species == 0: 
            return 7.5e-19 * (1. + np.sqrt(T / 1e5))**-1. * np.exp(-1.18e5 / T)
        if species == 1: 
            return 9.1e-27 * T**-0.1687 * (1. + np.sqrt(T / 1e5))**-1. * np.exp(-1.31e4 / T) * nion[1] / nabs[1]   # CONFUSION
        if species == 2: 
            return 5.54e-17 * T**-0.397 * (1. + np.sqrt(T / 1e5))**-1. * np.exp(-4.73e5 / T)    
        
    def RecombinationCoolingRate(self, species = 0, T = None):
        """
        Returns coefficient for cooling by recombination.  These are equations B4.2a, b, and d respectively
        from FK96.
        
            units: erg cm^3 / s
        """
        
        if self.Isothermal:
            return 0
        
        if species == 0: 
            return 6.5e-27 * T**0.5 * (T / 1e3)**-0.2 * (1.0 + (T / 1e6)**0.7)**-1.0
        if species == 1: 
            return 1.55e-26 * T**0.3647
        if species == 2: 
            return 3.48e-26 * np.sqrt(T) * (T / 1e3)**-0.2 * (1. + (T / 4e6)**0.7)**-1.
        
    def DielectricRecombinationCoolingRate(self, T):
        """
        Returns coefficient for cooling by dielectric recombination.  This is equation B4.2c from FK96.
        
            units: erg cm^3 / s
        """
        
        if self.Isothermal:
            return 0
        
        return 1.24e-13 * T**-1.5 * np.exp(-4.7e5 / T) * (1. + 0.3 * np.exp(-9.4e4 / T))
        
    #def IonizationRateCoefficientHeI(self, ncol, n_HI, n_HeI, x_HII, T, r, Lbol, indices):
    #    """
    #    Returns ionization rate coefficient for HeI, which we denote elsewhere as Gamma_HeI.  Includes photo 
    #    and secondary ionizations from fast electrons.  Unlike the hydrogen case, the collisional ionizations
    #    are included in the rate equation itself instead of a coefficient.
    #    
    #        units: 1 / s
    #    """                
    #    
    #    IonizationRate = Lbol * \
    #                     self.Interpolate.interp(indices, "PhotoIonizationRate1", ncol)
    #    
    #    if self.SecondaryIonization:
    #        IonizationRate += Lbol * \
    #                          self.esec.DepositionFraction(0.0, x_HII, channel = 2) * \
    #                          self.Interpolate.interp(indices, "SecondaryIonizationRateHeI1", ncol)
    #        
    #        IonizationRate += (n_HI / n_HeI) * Lbol * \
    #                          self.esec.DepositionFraction(0.0, x_HII, channel = 2) * \
    #                          self.Interpolate.interp(indices, "SecondaryIonizationRateHeI1", ncol) 
    #    
    #    if not self.PlaneParallelField: 
    #        IonizationRate /= (4. * np.pi * r**2)
    #    
    #    return IonizationRate
        
    #def IonizationRateCoefficientHeII(self, ncol, n_HI, n_HeI, x_HII, T, r, Lbol, indices):
    #    """
    #    Returns ionization rate coefficient for HeII, which we denote elsewhere as Gamma_HeII.  Includes photo 
    #    and secondary ionizations from fast electrons.  Unlike the hydrogen case, the collisional ionizations
    #    are included in the rate equation itself instead of a coefficient.  Note: TZ07 do not include secondary
    #    helium II ionizations, but I am going to.
    #    
    #        units: 1 / s
    #    """       
    #    
    #    IonizationRate = Lbol * \
    #                     self.Interpolate.interp(indices, "PhotoIonizationRate2", ncol) \
    #    
    #    if self.SecondaryIonization > 1:
    #        IonizationRate += Lbol * self.esec.DepositionFraction(0.0, x_HII, channel = 3) * \
    #            self.Interpolate.interp(indices, "SecondaryIonizationRate2", ncol)
    #                    
    #    if not self.PlaneParallelField: 
    #        IonizationRate /= (4. * np.pi * r**2)
    #    
    #    return IonizationRate
        
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
        

        
        
          