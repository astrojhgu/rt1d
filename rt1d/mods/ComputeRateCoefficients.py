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
        self.itabs = itabs
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
        
        if self.MultiSpecies:
            self.donors = np.arange(3)
        else:
            self.donors = np.array([0])
            
        self.mask = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])    
        
        self.sigma = np.zeros([3, len(self.rs.E)])
        for i in xrange(3):
            self.sigma[i] = PhotoIonizationCrossSection(self.rs.E, species = i)
        
    def ConstructArgs(self, args, indices, Lbol, r, ncol, T, dr, tau, t):
        """
        Make list of rate coefficients that we'll pass to solver.
        
            incoming args: [nabs, nion, n_H, n_He, n_e]
            outgoing args: [nabs, n_H, n_He, n_e, Gamma, gamma, Beta, alpha, k_H, zeta, eta, psi]
        """    
        
        nabs = args[0]
        nion = args[1]
        n_H = args[2]
        x_HII = (n_H - nabs[0]) / n_H
        n_He = args[3]
        n_e = args[4]
        
        Gamma = np.zeros(3)
        gamma = np.zeros(3)
        alpha = np.zeros(3)
        Beta = np.zeros(3)
        k_H = np.zeros(3)
        zeta = np.zeros(3)
        eta = np.zeros(3)
        psi = np.zeros(3)
        xi = np.zeros(3)
        Phi_N = np.zeros(3)
        Phi_N_dN = np.zeros(3)
        Psi_N = np.zeros(3)
        Psi_N_dN = np.zeros(3)
        

        # Derived quantities we'll need
        Vsh = self.ShellVolume(r, dr)
                        
        # Standard - integral tabulation
        if self.TabulateIntegrals:
            
            # Loop over species   
            for i in xrange(3):
                
                if not self.MultiSpecies and i > 0:
                    continue
                
                # A few quantities that are better to just compute once
                ncell = dr * nabs * self.mask[i]   
                nout = ncol + ncell
                indices_out = self.Interpolate.GetIndices3D(nout)
                
                if self.PhotonConserving:
                    A = Lbol / nabs[i] / Vsh
                else:
                    A = Lbol / 4. / np.pi / r**2
                                                           
                # Photo-Ionization - keep integral values too to be used in heating/secondary ionization
                Gamma[i], Phi_N[i], Phi_N_dN[i] = self.PhotoIonizationRate(species = i, indices_in = indices,  
                    Lbol = Lbol, r = r, ncol = ncol, nabs = nabs, tau = tau, dr = dr,
                    Vsh = Vsh, ncell = ncell, indices_out = indices_out, A = A, nout = nout)
                
                if self.CollisionalIonization:
                    Beta[i] = self.CollisionalIonizationRate(species = i, T = T, n_e = n_e)    
               
                # Recombination
                alpha[i] = self.RadiativeRecombinationRate(species = i, T = T)
                
                # Dielectric recombination
                if i == 2:
                    xi[i] = self.DielectricRecombinationRate(T = T)
                    
                # Heating/cooling                            
                if not self.Isothermal:
                    k_H[i], Psi_N[i], Psi_N_dN[i] = self.PhotoElectricHeatingRate(species = i, Lbol = Lbol, r = r, 
                        x_HII = x_HII, indices_in = indices, indices_out = indices_out, ncol = ncol, dr = dr, 
                        nout = nout, nabs = nabs, Phi_N = Phi_N[i], Phi_N_dN = Phi_N_dN[i], A = A)
                    zeta[i] = self.CollisionalIonizationCoolingRate(species = i, T = T)    
                    eta[i] = self.RecombinationCoolingRate(species = i, T = T) 
                    psi[i] = self.CollisionalExcitationCoolingRate(species = i, T = T, nabs = nabs, nion = nion)    
            
            # Secondary ionization - do separately to take advantage of already knowing Psi and Phi
            if self.SecondaryIonization:
                for i in xrange(3):
                    if not self.MultiSpecies and i > 0:
                        continue
                                            
                    for j in self.donors:
                        gamma[i] += self.SecondaryIonizationRate(Psi_N = Psi_N[j], Psi_N_dN = Psi_N_dN[j],
                            Phi_N = Phi_N[j], Phi_N_dN = Phi_N_dN[j], 
                            Lbol = Lbol, r = r, ncol = ncol, nabs = nabs, tau = tau, dr = dr,
                            species = i, donor_species = j, x_HII = x_HII)
                                
                    gamma[i] *= (A / E_th[i] / erg_per_ev)        
                                    
        # Only the photon-conserving algorithm is capable of this                                                  
        else:
               
            # Loop over species   
            for i in xrange(3):
                
                if not self.MultiSpecies and i > 0:
                    continue
                                
                # Loop over energy groups
                Gamma_E = np.zeros_like(self.rs.E)
                for j, E in enumerate(self.rs.E):
                                    
                    # Photo-ionization by *this* energy group
                    Gamma_E[j], tmp, tmp = self.PhotoIonizationRate(E = E, Qdot = self.rs.IonizingPhotonLuminosity(t = t, i = j), \
                        ncol = ncol, nabs = nabs, r = r, dr = dr, species = i, tau = tau, Lbol = Lbol, bin = j)
                        
                    # Total photo-ionization tally
                    Gamma[i] += Gamma_E[j]
                    
                    # Energy in photo-electrons due to ionizations by *this* energy group
                    ee = Gamma_E[j] * (E - E_th[i]) * erg_per_ev                    
                        
                    if self.SecondaryIonization:
                        
                        # Ionizations of species k by photoelectrons from species i
                        for k in xrange(3):
                            if not self.MultiSpecies and k > 0:
                                continue
                            
                            # If these photo-electrons dont have enough energy to ionize species k, continue    
                            if (E - E_th[i]) <= E_th[k]:
                                continue    
                            
                            gamma[k] += ee * self.esec.DepositionFraction(E = E, xi = x_HII, channel = 1 + k) * \
                                (nabs[i] / nabs[k]) / \
                                (E_th[i] * erg_per_ev)
                                                                                                        
                    if self.Isothermal:
                        continue
                            
                    # Heating rate coefficient        
                    k_H[i] += ee * self.esec.DepositionFraction(E = E, xi = x_HII, channel = 0)    
                                    
                # Recombination
                alpha[i] = self.RadiativeRecombinationRate(species = i, T = T)
                
                # Dielectric recombination
                if i == 2:
                    xi[i] = self.DielectricRecombinationRate(T = T)
                
                if self.CollisionalIonization:
                    Beta[i] = self.CollisionalIonizationRate(species = i, T = T, n_e = n_e)
                                
                if self.Isothermal:
                    continue
                
                zeta[i] += self.CollisionalIonizationCoolingRate(species = i, T = T)  
                eta[i] = self.RecombinationCoolingRate(species = i, T = T) 
                psi[i] = self.CollisionalExcitationCoolingRate(species = i, T = T, nabs = nabs, nion = nion)               

        return [Gamma, gamma, Beta, alpha, k_H, zeta, eta, psi, xi]

    def PhotoIonizationRate(self, species = None, E = None, Qdot = None, Lbol = None, 
        ncol = None, n_e = None, nabs = None, x_HII = None, indices_out = None,
        T = None, r = None, dr = None, indices_in = None, Vsh = None, ncell = None, nout = None,
        A = 1.0, tau = None, bin = 0):
        """
        Returns photo-ionization rate coefficient which we denote elsewhere as Gamma.  
        
            returns IonizationRate: [IonizationRate] = 1 / s
            
            Inputs:
            E = Energy of photons in ray (eV)
            Qdot = Photon luminosity of source for this ray (s^-1)
            Lbol = Bolometric luminosity of source (erg/s)
            
                        
            can calculate all shell volumes at once too
            
        """     
                  
        Phi_N = None
        Phi_N_dN = None
                                        
        if self.TabulateIntegrals:
            if self.PhotonConserving:
                Phi_N = self.Interpolate.interp(indices_in, "PhotoIonizationRate%i" % species, ncol)
                Phi_N_dN = self.Interpolate.interp(indices_out, "PhotoIonizationRate%i" % species, nout)
                IonizationRate = Phi_N - Phi_N_dN
            else:                
                Phi_N = self.Interpolate.interp(indices_in, "PhotoIonizationRate%i" % species, ncol)       
                IonizationRate = Phi_N
            
        else:
            tau_E = ncol[species] * self.sigma[species][bin]       # Optical depth up until this cell
            tau_c = dr * nabs[species] * self.sigma[species][bin]  # Optical depth of this cell
            Q0 = Qdot * np.exp(-tau_E)                             # number of photons entering cell per sec
            dQ = Q0 * (1. - np.exp(-tau_c))                        # number of photons absorbed in cell per sec
            
            IonizationRate = dQ / nabs[species] / self.ShellVolume(r, dr)    # ionizations / sec / hydrogen atom
    
        #print IonizationRate, ncol, nout
            
        return A * IonizationRate, Phi_N, Phi_N_dN
        
    def PhotoElectricHeatingRate(self, species = None, E = None, Qdot = None, Lbol = None, 
        ncol = None, n_e = None, nabs = None, x_HII = None, 
        T = None, r = None, dr = None, dt = None, indices_in = None, indices_out = None,
        Phi_N = None, Phi_N_dN = None, nout = None, A = None):
        """
        Photo-electric heating rate coefficient due to photo-electrons previously 
        bound to `species.'  If this method is called, it means TabulateIntegrals = 1.
        """
        
        Psi_N = None
        Psi_N_dN = None
        
        if self.PhotonConserving:
            Psi_N = self.Interpolate.interp(indices_in, "ElectronHeatingRate%i" % species, ncol)
            Psi_N_dN = self.Interpolate.interp(indices_out, "ElectronHeatingRate%i" % species, nout)
                        
            heat = (Psi_N - Psi_N_dN - E_th[species] * erg_per_ev * (Phi_N - Phi_N_dN))
                   
        else:
            Psi_N = self.Interpolate.interp(indices_in, "ElectronHeatingRate%i" % species, ncol)
            heat = Psi_N

        return A * self.esec.DepositionFraction(0.0, x_HII, channel = 0) * heat, Psi_N, Psi_N_dN    
        
    def SecondaryIonizationRate(self, species = 0, donor_species = 0, E = None, Qdot = None, Lbol = None, 
        ncol = None, n_e = None, nabs = None, x_HII = None, tau = None,
        T = None, r = None, dr = None, dt = None, Psi_N = None, Psi_N_dN = None,
        Phi_N = None, Phi_N_dN = None):
        """
        Secondary ionization rate which we denote elsewhere as gamma (note little g).
        
            species = species being ionized by photo-electron
            donor_species = species the photo-electron came from
            
        If this routine is called, it means TabulateIntegrals = 1.    
        """    
        
        if self.PhotonConserving:
            IonizationRate = (Psi_N - Psi_N_dN - E_th[donor_species] * erg_per_ev * (Phi_N - Phi_N_dN))        
        else:            
            IonizationRate = (Psi_N - E_th[donor_species] * erg_per_ev * Phi_N)   
             
        # Normalization will be applied in ConstructArgs                                                                                                                      
        return IonizationRate * \
            self.esec.DepositionFraction(E = E, xi = x_HII, channel = species + 1) * \
            (nabs[donor_species] / nabs[species])
        
    def CollisionalIonizationRate(self, species = None, n_e = None, T = None):
        """
        Collisional ionization rate which we denote elsewhere as Beta.
        """    
        
        if species == 0:  
            return n_e * 5.85e-11 * np.sqrt(T) * (1. + np.sqrt(T / 1.e5))**-1. * np.exp(-1.578e5 / T)    
          
        if species == 1:    
            return n_e * 2.38e-11 * np.sqrt(T) * (1. + np.sqrt(T / 1.e5))**-1. * np.exp(-2.853e5 / T) 
        
        if species == 2:
            return n_e * 5.68e-12 * np.sqrt(T) * (1. + np.sqrt(T / 1.e5))**-1. * np.exp(-6.315e5 / T)     
        
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
        
    def CollisionalIonizationCoolingRate(self, species = 0, T = None):
        """
        Returns coefficient for cooling by collisional ionization.  These are equations B4.1a, b, and d respectively
        from FK96.
        
            units: erg cm^3 / s
        """
        
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
        
        return 1.24e-13 * T**-1.5 * np.exp(-4.7e5 / T) * (1. + 0.3 * np.exp(-9.4e4 / T))
        
    def ShellVolume(self, r, dr):
        """
        Return volume of shell at distance r, thickness dr.
        """
        
        return 4. * np.pi * ((r + dr)**3 - r**3) / 3.    
        
          