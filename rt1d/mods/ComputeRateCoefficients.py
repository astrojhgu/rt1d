"""

ComputeRateCoefficients.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Tue Jan 17 16:57:48 2012

Description: 

"""

import numpy as np

from .Constants import erg_per_ev, k_B, c, m_e, sigma_SB, sigma_T 
from .Cosmology import Cosmology
from .Interpolate import Interpolate
from .RadiationSource import RadiationSource
from .SecondaryElectrons import SecondaryElectrons
from .ComputeCrossSections import *
from scipy.integrate import quad

E_th = np.array([13.6, 24.6, 54.4])

class RateCoefficients:
    def __init__(self, pf, rs = None, grid = None):
        self.pf = pf        
        self.rs = rs
        self.cosm = Cosmology(pf)
        self.esec = SecondaryElectrons(pf)
        
        self.mask = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])    
        if pf['MultiSpecies']:
            self.donors = np.arange(3)
        else:
            self.donors = np.array([0]) 
        
        self.Vsh = 4. * np.pi * ((grid.r + grid.dx)**3 - grid.r**3) / 3.
        
        self.smallcol = pf['OpticallyThinColumn']
                                    
    def ConstructArgs(self, args, indices, Lbol, r, ncol, T, dr, t, z, cell):
        """
        Make list of rate coefficients that we'll pass to solver.
        
            incoming args: [nabs, nion, n_H, n_He, n_e]
            outgoing args: [nabs, n_H, n_He, n_e, Gamma, gamma, Beta, alpha, k_H, zeta, eta, psi, xi, omega, theta]
            
            ***REMEMBER: ncol is really np.log10(ncol)!
            
        """    
                
        nabs = args[0]
        nion = args[1]
        n_H = args[2]
        x_HII = (n_H - nabs[0]) / n_H
        n_He = args[3]
        n_e = args[4]
        
        Gamma = np.zeros([3, self.rs.Ns])
        gamma = np.zeros([3, 3, self.rs.Ns])
        alpha = np.zeros(3)
        Beta = np.zeros(3)
        k_H = np.zeros([3, self.rs.Ns])
        zeta = np.zeros(3)
        eta = np.zeros(3)
        psi = np.zeros(3)
        xi = np.zeros(3)
        omega = np.zeros(3)
        theta = 0
        Phi_N = np.zeros(3)
        Phi_N_dN = np.zeros(3)
        Psi_N = np.zeros(3)
        Psi_N_dN = np.zeros(3)
        
        # Derived quantities we'll need
        Vsh = self.Vsh[cell]        
        Ncell = dr * nabs          
        logNcell = np.log10(Ncell)
        
        # Loop over radiation sources                            
        for rs, source in enumerate(self.rs.all_sources):
            
            ON = source.SourceOn(t)    
                                                                
            # Set normalization constant for each species
            if self.pf['PlaneParallelField']:
                if self.pf['PhotonConserving']:
                    A = Lbol[rs] / Ncell
                else:
                    A = 3 * [Lbol[rs]]
            else:    
                if self.pf['PhotonConserving']:
                    A = Lbol[rs] / nabs / Vsh
                else:
                    A = 3 * [Lbol[rs] / 4. / np.pi / r**2]
                                                                                                         
            # Check to see if we're in the small tau limit      
            small_tau = [False, False, False]      
            if self.pf['AllowSmallTauApprox']:      
                tau_small = (ncol[0] <= self.smallcol[0]) & (ncol[1] <= self.smallcol[1]) \
                    & (ncol[2] <= self.smallcol[2])  
                    
                small_tau = [tau_small, tau_small, tau_small]
                for i in xrange(3): 
                    if i > 0 and not self.pf['MultiSpecies']:
                        continue
                               
                    small_tau[i] &= (logNcell[i] <= self.smallcol[i])
                                                    
            # Standard - integral tabulation
            if not self.pf['DiscreteSpectrum'] or self.pf['ForceIntegralTabulation']:
             
                # Loop over species   
                for i in xrange(3):
                                    
                    if not self.pf['MultiSpecies'] and i > 0:
                        continue
                                    
                    # A few quantities that are better to just compute once   
                    nout = np.log10(10**ncol + Ncell * self.mask[i])
                    indices_out = source.Interpolate.GetIndices([nout[0], nout[1], nout[2], np.log10(x_HII), t])
                                                                                                                                                 
                    # Photo-Ionization - keep integral values too to be used in heating/secondary ionization
                    if small_tau[i]:
                        if self.pf['PlaneParallelField']:
                            Gamma[i][rs] = int(ON) * source.Gamma_const[i] * Lbol[rs]
                        else:    
                            Gamma[i][rs] = int(ON) * source.Gamma_const[i] * Lbol[rs] / 4. / np.pi / r**2
                    else:
                        Gamma[i][rs], Phi_N[i], Phi_N_dN[i] = self.PhotoIonizationRate(rs = source, 
                            species = i, indices_in = indices[rs], ON = ON,
                            Lbol = Lbol[rs], r = r, ncol = ncol, nabs = nabs, dr = dr, x_HII = x_HII,
                            Vsh = Vsh, Ncell = Ncell, indices_out = indices_out, A = A[i], nout = nout, t = t)
                    
                    # Source independent terms
                    if rs == 0:                                             
                        if self.pf['CollisionalIonization']:
                            Beta[i] = self.CollisionalIonizationRate(species = i, T = T, n_e = n_e)
                        
                        # Recombination
                        alpha[i] = self.RadiativeRecombinationRate(species = i, T = T)
                        
                        # Dielectric recombination
                        if i == 1:
                            xi[i] = self.DielectricRecombinationRate(T = T)
                            omega[i] = self.DielectricRecombinationCoolingRate(T = T)
                        
                    # Heating/cooling                            
                    if not self.pf['Isothermal']:
                        if small_tau[i]:
                            k_H[i][rs] = Gamma[i][rs] * source.Heat_const[i]
                        else:
                            k_H[i][rs], Psi_N[i], Psi_N_dN[i] = self.PhotoElectricHeatingRate(rs = source, 
                                species = i, Lbol = Lbol[rs], r = r, ON = ON,
                                x_HII = x_HII, indices_in = indices[rs], indices_out = indices_out, ncol = ncol, dr = dr, 
                                nout = nout, nabs = nabs, Phi_N = Phi_N[i], Phi_N_dN = Phi_N_dN[i], A = A[i], t = t)
                        
                        if rs == 0:
                            zeta[i] = self.CollisionalIonizationCoolingRate(species = i, T = T)    
                            eta[i] = self.RecombinationCoolingRate(species = i, T = T) 
                            psi[i] = self.CollisionalExcitationCoolingRate(species = i, T = T, nabs = nabs, nion = nion)    
                
                # Secondary ionization - do separately to take advantage of already knowing Psi and Phi
                # Unless SecondaryIonization = 2 -- then knowing Phi and Psi for Gamma won't matter
                if self.pf['SecondaryIonization']:
                    
                    for i in xrange(3):
                        if not self.pf['MultiSpecies'] and i > 0:
                            continue  
                        
                        # Ionizing species i with electrons from species j                        
                        for j in self.donors:
                            if not self.pf['MultiSpecies'] and j > 0:
                                continue
                                
                            if small_tau[j]:
                                gamma[i][j][rs] += Gamma[j][rs] * source.gamma_const[i][j]
                            else:        
                                gamma[i][j][rs] += self.SecondaryIonizationRate(rs = source, 
                                    Psi_N = Psi_N[j], Psi_N_dN = Psi_N_dN[j], ON = ON,
                                    Phi_N = Phi_N[j], Phi_N_dN = Phi_N_dN[j], t = t,
                                    Lbol = Lbol[rs], r = r, ncol = ncol, nabs = nabs, dr = dr,
                                    species = i, donor_species = j, x_HII = x_HII, A = A[j],
                                    indices_in = indices[rs], indices_out = indices_out, nout = nout)
                                        
                        gamma[i,:,rs] /= (E_th[i] * erg_per_ev)        
                                                            
            # Only the photon-conserving algorithm is capable of this - though in
            # the future, the discrete NPC solver could do this if we wanted.                                         
            else:
                
                Qdot = np.zeros_like(source.E)
                tau_c = np.zeros([len(source.E), 3])
                for i in xrange(source.Nfg):
                    Qdot[i] = source.IonizingPhotonLuminosity(t = t, bin = i)
                    for j in xrange(3):            
                        tau_c[i][j] = dr * nabs[j] * source.sigma[j][i]
                                                                                                 
                # Loop over species  
                for i in xrange(3):
            
                    if not self.pf['MultiSpecies'] and i > 0:
                        continue
                                                        
                    # Loop over energy groups
                    Gamma_E = np.zeros_like(source.E)
                    for j, E in enumerate(source.E):
                        
                        if E < E_th[i]:
                            continue
                        
                        fheat = self.esec.DepositionFraction(E = E, xHII = x_HII, channel = 0)
                        
                        # Optical depth up to this cell at energy E
                        tau_E = np.sum(10**ncol * source.sigma[0:,j])
                                                                        
                        # Photo-ionization by *this* energy group
                        Gamma_E[j], tmp, tmp = self.PhotoIonizationRate(rs = source, E = E, 
                            Qdot = Qdot[j], nabs = nabs, dr = dr, species = i, t = t, ON = ON,
                            Vsh = Vsh, tau_E = tau_E, tau_c = tau_c[j][i], A = 1.0)
                            
                        Gamma_E[j] = Gamma_E[j] 
                                                    
                        # Total photo-ionization tally
                        Gamma[i] += Gamma_E[j]
                        
                        # Total energy deposition rate per atom i via photo-electrons 
                        # due to ionizations by *this* energy group. 
                        ee = Gamma_E[j] * (E - E_th[i]) * erg_per_ev 
                            
                        if self.pf['SecondaryIonization']:
                            
                            # Ionizations of species k by photoelectrons from species i
                            # Neglect HeII until somebody figures out how that works
                            for k in xrange(2):
                                if not self.pf['MultiSpecies'] and k > 0:
                                    continue
                                
                                # If these photo-electrons dont have enough energy to ionize species k, continue    
                                if (E - E_th[i]) < E_th[k]:
                                    continue    
                                
                                fion = self.esec.DepositionFraction(E = E, xHII = x_HII, channel = k + 1)
                                
                                # (This k) = i from paper, and (this i) = j from paper
                                gamma[k][i] += ee * fion / (E_th[k] * erg_per_ev)
                                                                                                              
                        if self.pf['Isothermal']:
                            continue                           
                                                            
                        # Heating rate coefficient        
                        k_H[i] += ee * fheat
                                      
                    if rs == 0:                                                                    
                        # Recombination
                        alpha[i] = self.RadiativeRecombinationRate(species = i, T = T)
                        
                        # Dielectric recombination
                        if i == 1:
                            xi[i] = self.DielectricRecombinationRate(T = T)
                            omega[i] = self.DielectricRecombinationCoolingRate(T = T)
                        
                        if self.pf['CollisionalIonization']:
                            Beta[i] = self.CollisionalIonizationRate(species = i, T = T, n_e = n_e)
                                        
                        if self.pf['Isothermal']:
                            continue
                        
                        zeta[i] += self.CollisionalIonizationCoolingRate(species = i, T = T)  
                        eta[i] = self.RecombinationCoolingRate(species = i, T = T) 
                        psi[i] = self.CollisionalExcitationCoolingRate(species = i, T = T, nabs = nabs, nion = nion)     
        
        hubble = 0
        compton = 0                                 
        if self.pf['CosmologicalExpansion']:
            hubble = self.HubbleCoolingRate(z)
            compton = self.ComptonHeatingRate(z, n_H, n_He, n_e, x_HII, T)
                                                                                                                      
        return [Gamma, gamma, Beta, alpha, k_H, zeta, eta, psi, xi, omega, hubble, compton]

    def PhotoIonizationRate(self, rs = None, species = None, E = None, Qdot = None, Lbol = None, 
        ncol = None, n_e = None, nabs = None, x_HII = None, indices_out = None,
        T = None, r = None, dr = None, indices_in = None, Vsh = None, Ncell = None, nout = None,
        A = None, tau_E = None, tau_c = None, t = None, ON = None):
        """
        Returns photo-ionization rate coefficient which we denote elsewhere as Gamma.  
        
            [IonizationRate] = 1 / s
            
            Inputs:
            E = Energy of photons in ray (eV)
            Qdot = Photon luminosity of source for this ray (s^-1)
            Lbol = Bolometric luminosity of source (erg/s)
                        
        """     
                                  
        if not ON:
            return 0.0, 0, 0
                                  
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
        else:
            Q0 = Qdot * np.exp(-tau_E)                 # number of photons entering cell per sec
            dQ = Q0 * (1. - np.exp(-tau_c))            # number of photons absorbed in cell per sec
            IonizationRate = dQ / nabs[species] / Vsh  # ionizations / sec / atom        
                          
        return A * IonizationRate, Phi_N, Phi_N_dN
        
    def PhotoElectricHeatingRate(self, rs = None, species = None, E = None, Qdot = None, Lbol = None, 
        ncol = None, n_e = None, nabs = None, x_HII = None, ON = None,
        T = None, r = None, dr = None, dt = None, indices_in = None, indices_out = None,
        Phi_N = None, Phi_N_dN = None, nout = None, A = None, t = None):
        """
        Photo-electric heating rate coefficient due to photo-electrons previously 
        bound to `species.'  If this method is called, it means TabulateIntegrals = 1.
        """
        
        if not ON:
            return 0.0, 0, 0 
        
        Psi_N = Psi_N_dN = None
        
        if self.esec.Method < 2:
            if rs.pf['PhotonConserving']:
                Psi_N = rs.Interpolate.interp(indices_in, "logPsi%i" % species, 
                    [ncol[0], ncol[1], ncol[2], x_HII, t])
                Psi_N_dN = rs.Interpolate.interp(indices_out, "logPsi%i" % species, 
                    [nout[0], nout[1], nout[2], x_HII, t])
                heat = (Psi_N - Psi_N_dN - E_th[species] * erg_per_ev * (Phi_N - Phi_N_dN))                   
            else:
                Psi_N = rs.Interpolate.interp(indices_in, "logPsi%i" % species, 
                    [ncol[0], ncol[1], ncol[2], x_HII, t])
                heat = Psi_N
                
            return A * self.esec.DepositionFraction(0.0, x_HII, channel = 0) * heat, Psi_N, Psi_N_dN    
                
        else:
            if rs.pf['PhotonConserving']:
                x_HII = np.log10(x_HII)
                Psi_N = rs.Interpolate.interp(indices_in, "logPsiHat%i" % species, 
                    [ncol[0], ncol[1], ncol[2], x_HII, t])
                Psi_N_dN = rs.Interpolate.interp(indices_out, "logPsiHat%i" % species, 
                    [nout[0], nout[1], nout[2], x_HII, t])
                Phi_N = rs.Interpolate.interp(indices_in, "logPhiHat%i" % species, 
                    [ncol[0], ncol[1], ncol[2], x_HII, t])
                Phi_N_dN = rs.Interpolate.interp(indices_out, "logPhiHat%i" % species, 
                    [nout[0], nout[1], nout[2], x_HII, t])
                                
                heat = (Psi_N - Psi_N_dN - E_th[species] * erg_per_ev * (Phi_N - Phi_N_dN))                   
            else:
                Psi_N = rs.Interpolate.interp(indices_in, "logPsiHat%i" % species,
                    [ncol[0], ncol[1], ncol[2], x_HII, t])
                heat = Psi_N                    
                                    
            return A * heat, None, None            
        
    def SecondaryIonizationRate(self, rs = None, species = 0, donor_species = 0, E = None, Qdot = None, Lbol = None, 
        ncol = None, n_e = None, nabs = None, x_HII = None, tau = None,
        T = None, r = None, dr = None, dt = None, Psi_N = None, Psi_N_dN = None,
        Phi_N = None, Phi_N_dN = None, A = None, t = None, ON = None,
        indices_in = None, indices_out = None, nout = None):
        """
        Secondary ionization rate which we denote elsewhere as gamma (note little g).
        
            species = species being ionized by photo-electron
            donor_species = species the photo-electron came from
            
        If this routine is called, it means TabulateIntegrals = 1.    
        """    
        
        if not ON:
            return 0.0, 0, 0
            
        if self.esec.Method < 2:    
            if rs.pf['PhotonConserving']:
                IonizationRate = (Psi_N - Psi_N_dN - E_th[donor_species] * erg_per_ev * (Phi_N - Phi_N_dN))        
            else:            
                IonizationRate = (Psi_N - E_th[donor_species] * erg_per_ev * Phi_N)   
        
            # Normalization will be applied in ConstructArgs                                                                                                                      
            return A * IonizationRate * \
                self.esec.DepositionFraction(E = E, xHII = x_HII, channel = species + 1)
        
        else:
            if rs.pf['PhotonConserving']:
                x_HII = np.log10(x_HII)                
                Psi_N = rs.Interpolate.interp(indices_in, "logPsiWiggle%i%i" % (species, donor_species), 
                    [ncol[0], ncol[1], ncol[2], x_HII, t])
                Psi_N_dN = rs.Interpolate.interp(indices_out, "logPsiWiggle%i%i" % (species, donor_species), 
                    [nout[0], nout[1], nout[2], x_HII, t])
                Phi_N = rs.Interpolate.interp(indices_in, "logPhiWiggle%i%i" % (species, donor_species), 
                    [ncol[0], ncol[1], ncol[2], x_HII, t])
                Phi_N_dN = rs.Interpolate.interp(indices_out, "logPhiWiggle%i%i" % (species, donor_species), 
                    [nout[0], nout[1], nout[2], x_HII, t])
                IonizationRate = (Psi_N - Psi_N_dN - E_th[donor_species] * erg_per_ev * (Phi_N - Phi_N_dN))        
            else:            
                IonizationRate = (Psi_N - E_th[donor_species] * erg_per_ev * Phi_N)   
        
            return A * IonizationRate
        
    def CollisionalIonizationRate(self, species = None, n_e = None, T = None):
        """
        Collisional ionization rate which we denote elsewhere as Beta.
        """    
        
        if species == 0:  
            return 5.85e-11 * np.sqrt(T) * (1. + np.sqrt(T / 1.e5))**-1. * np.exp(-1.578e5 / T)    
          
        if species == 1:    
            return 2.38e-11 * np.sqrt(T) * (1. + np.sqrt(T / 1.e5))**-1. * np.exp(-2.853e5 / T) 
        
        if species == 2:
            return 5.68e-12 * np.sqrt(T) * (1. + np.sqrt(T / 1.e5))**-1. * np.exp(-6.315e5 / T)     
        
    def RadiativeRecombinationRate(self, species = 0, T = None):
        """
        Coefficient for radiative recombination.  Here, species = 0, 1, 2
        refers to HII, HeII, and HeIII.
        """
        
        if self.pf['RecombinationMethod'] == 'A':
            if species == 0:
                return 6.28e-11 * T**-0.5 * (T / 1e3)**-0.2 * (1. + (T / 1e6)**0.7)**-1.
            elif species == 1:
                return 1.5e-10 * T**-0.6353
            elif species == 2:
                return 3.36e-10 * T**-0.5 * (T / 1e3)**-0.2 * (1. + (T / 4e6)**0.7)**-1.
        elif self.pf['RecombinationMethod'] == 'B':
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
        else:
            print 'Unrecognized RecombinationMethod.  Should be A or B.'
            return 0.0          
        
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
            return 6.5e-27 * np.sqrt(T) * (T / 1e3)**-0.2 * (1.0 + (T / 1e6)**0.7)**-1.0
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
        
    def HubbleCoolingRate(self, z):
        """
        Just the Hubble parameter. 
        """
        
        return self.cosm.HubbleParameter(z)
        
    def ComptonHeatingRate(self, z, nH, nHe, ne, xHII, Tk):
        """
        Compton heating rate due to electron-CMB scattering.
        """    
        
        if z > self.cosm.zdec:
            Tcmb = self.cosm.TCMB(z)
            ucmb = sigma_SB * Tcmb**4. / 4. / np.pi / c
            
            return 4. * sigma_T * ne * ucmb * k_B * (Tcmb - Tk) / m_e / c
            #return xHII * k_B * (nH + nHe + ne) * 4. * sigma_T * ucmb * \
            #    (Tcmb - Tk) / (1. + self.cosm.y + xHII) / m_e / c   
        else:
            return 0.0
            # Implement source-dependent compton heating        
    
    def ShellVolume(self, r, dr):
        """
        Return volume of shell at distance r, thickness dr.
        """
        
        return 4. * np.pi * ((r + dr)**3 - r**3) / 3.    
        
          