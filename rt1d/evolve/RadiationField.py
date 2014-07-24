"""

RadiationField.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Dec 31 11:16:59 2012

Description:

"""

import copy
import numpy as np
from ..util import parse_kwargs
from ..physics.SecondaryElectrons import *
from ..physics.Constants import erg_per_ev, E_LyA, ev_per_hz

class RadiationField:
    def __init__(self, grid, sources, **kwargs):
        self.pf = parse_kwargs(**kwargs)
        self.grid = grid
        self.srcs = sources
        self.esec = SecondaryElectrons(method=self.pf['secondary_ionization'])
                
        if self.srcs is not None:
            self._initialize()
        
    def _initialize(self):
        self.Ns = len(self.srcs)
                
        # See if all sources are diffuse
        self.all_diffuse = 1
        for src in self.srcs:
            self.all_diffuse *= int(src.SourcePars['type'] == 'diffuse')
            
        if self.all_diffuse:
            return
        
        self.E_th = {}
        for absorber in self.grid.absorbers:
            self.E_th[absorber] = self.grid.ioniz_thresholds[absorber]
        
        # Array of cross-sections to match grid size 
        self.sigma = []
        for src in self.srcs:
            if src.continuous:
                self.sigma.append(None)
                continue

            self.sigma.append((np.ones([self.grid.dims, src.Nfreq]) \
                   * src.sigma).T)
        
        # Calculate correction to normalization factor if plane_parallel
        if self.pf['optically_thin']:
            if self.pf['plane_parallel']:
                self.pp_corr = 4. * np.pi * self.grid.r_mid**2
            else:
                self.pp_corr = 1.0
        else:
            if self.pf['photon_conserving']:
                self.pp_corr = self.grid.Vsh / self.grid.dr
            else:
                self.A_npc = source.Lbol / 4. / np.pi / self.grid.r_mid**2
                self.pp_corr = 4. * np.pi * self.grid.r_mid**2
                                        
    @property
    def finite_c(self):
        if self.pf['infinite_c']:
            return False
        return True            
            
    def SourceDependentCoefficients(self, data, t, z=None, **kwargs):
        """
        Compute rate coefficients for photo-ionization, secondary ionization, 
        and photo-heating.
        """
        
        self.k_H = np.array(self.Ns*[np.zeros_like(self.grid.zeros_grid_x_absorbers)])
        self.Gamma = np.array(self.Ns*[np.zeros_like(self.grid.zeros_grid_x_absorbers)])
        self.gamma = np.array(self.Ns*[np.zeros_like(self.grid.zeros_grid_x_absorbers2)])

        if self.pf['approx_lya']:
            self.Ja = [None] * self.Ns
        else:
            self.Ja = np.array(self.Ns * [np.zeros(self.grid.dims)])

        # H2 dissociation
        #self.kdiss = np.array(self.Ns*[np.zeros_like(self.grid.zeros_grid_x_absorbers)])

        # Column density to cells (N) and of cells (Nc)
        if not self.pf['optically_thin'] or self.all_diffuse:
            self.N, self.logN, self.Nc = self.grid.ColumnDensity(data)
            
            # Column densities (of all absorbers) sorted by cell 
            # (i.e. an array with shape = grid cells x # of absorbers)
            self.N_by_cell = np.zeros([self.grid.dims, len(self.grid.absorbers)])
            self.Nc_by_cell = np.zeros([self.grid.dims, len(self.grid.absorbers)])
            for i, absorber in enumerate(self.grid.absorbers):
                self.N_by_cell[...,i] = self.N[absorber]
                self.Nc_by_cell[...,i] = self.Nc[absorber]
            
            self.logN_by_cell = np.log10(self.N_by_cell)
            self.logNc_by_cell = np.log10(self.N_by_cell)
            
            # Number densities
            self.n = {}
            for absorber in self.grid.absorbers:
                self.n[absorber] = data[absorber] * self.grid.x_to_n[absorber]
                          
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
                          
        # Loop over sources
        for h, src in enumerate(self.srcs):        
            if not src.SourceOn(t):
                continue
            
            # "Diffuse" sources have parameterized rates    
            if src.SourcePars['type'] == 'diffuse':
                
                if 'Gamma_HI' in kwargs:
                    
                    # These if/else's are for glorb!
                    # Have to be sure we don't double count ionization/heat...
                    if src.pf['is_ion_src_HII']:
                        self.Gamma[h] = kwargs['Gamma_HI']
                    else:
                        self.Gamma[h] = 0.0
                else:
                    self.Gamma[h] = src.ionization_rate(z, **kwargs)
                
                if 'gamma_HI' in kwargs:
                    if src.pf['is_ion_src_igm']:
                        self.gamma[h] = kwargs['gamma_HI']
                    else:
                        self.gamma[h] = 0.0
                else:
                    self.gamma[h] = src.secondary_ionization_rate(z, **kwargs)
                
                if 'epsilon_X' in kwargs:
                    if src.pf['is_xray_src']:
                        self.k_H[h] = kwargs['epsilon_X']
                    else:
                        self.k_H[h] = 0.0
                else:
                    self.k_H[h] = src.heating_rate(z, **kwargs)

                continue    
                
            self.h = h
            self.src = src
                
            # If we're operating under the optically thin assumption, 
            # return pre-computed source-dependent values.    
            if self.pf['optically_thin']:
                self.tau_tot = np.zeros(self.grid.dims) # by definition
                self.Gamma[h] = src.Gamma_bar * self.pp_corr
                self.k_H[h] = src.Heat_bar * self.pp_corr
                self.gamma[h] = src.gamma_bar * self.pp_corr
                continue

            # Normalizations
            self.A = {}
            for absorber in self.grid.absorbers:                        
                if self.pf['photon_conserving']:
                    self.A[absorber] = self.src.Lbol \
                        / self.n[absorber] / self.grid.Vsh
                else:
                    self.A[absorber] = self.A_npc
                    
                # Correct normalizations if radiation field is plane-parallel
                if self.pf['plane_parallel']:
                    self.A[absorber] = self.A[absorber] * self.pp_corr
                                
            """
            For sources with discrete SEDs.
            """
            if self.src.discrete:
            
                # Loop over absorbing species
                for i, absorber in enumerate(self.grid.absorbers):
                                    
                    # Discrete spectrum (multi-freq approach)
                    if self.src.multi_freq:
                        self.Gamma[h,:,i], self.gamma[h,:,i], self.k_H[h,:,i] = \
                            self.MultiFreqCoefficients(data, absorber)
                    
                    # Discrete spectrum (multi-grp approach)
                    elif self.src.multi_group:
                        pass
                
                continue
                
            """
            For sources with continuous SEDs.
            """
            
            # This could be post-processed, but eventually may be more
            # sophisticated
            if self.pf['approx_lya']:
                self.Ja = None
            else:
                self.Ja[h] = src.Spectrum(E_LyA) * ev_per_hz \
                    * src.Lbol / 4. / np.pi / self.grid.r_mid**2 \
                    / E_LyA / erg_per_ev 
            
            # Initialize some arrays/dicts
            self.PhiN = {}
            self.PhiNdN = {}
            self.fheat = 1.0
            self.fion = dict([(absorber, 1.0) for absorber in self.grid.absorbers])
            
            if not self.pf['isothermal'] and self.pf['secondary_ionization'] < 2:
                self.PsiN = {}
                self.PsiNdN = {}
                self.fheat = self.esec.DepositionFraction(data['h_2'], 
                    channel='heat')
                
            self.logx = None            
            if self.pf['secondary_ionization'] > 1:
                
                self.logx = np.log10(data['h_2'])
                
                self.PhiWiggleN = {}
                self.PhiWiggleNdN = {}
                self.PhiHatN = {}
                self.PhiHatNdN = {}
                self.PsiWiggleN = {}
                self.PsiWiggleNdN = {}
                self.PsiHatN = {}
                self.PsiHatNdN = {}
                
                for absorber in self.grid.absorbers:
                    self.PhiWiggleN[absorber] = {}
                    self.PhiWiggleNdN[absorber] = {}
                    self.PsiWiggleN[absorber] = {}
                    self.PsiWiggleNdN[absorber] = {}
                
            else:
                self.fion = {}
                for absorber in self.grid.absorbers:
                    self.fion[absorber] = \
                        self.esec.DepositionFraction(xHII=data['h_2'], 
                            channel=absorber)
                            
            # Loop over absorbing species, compute tabulated quantities
            for i, absorber in enumerate(self.grid.absorbers):
                                           
                self.PhiN[absorber] = \
                    10**self.src.tables["logPhi_%s" % absorber](self.logN_by_cell,
                    self.logx, t)
                
                if (not self.pf['isothermal']) and (self.pf['secondary_ionization'] < 2):
                    self.PsiN[absorber] = \
                        10**self.src.tables["logPsi_%s" % absorber](self.logN_by_cell,
                        self.logx, t)
                    
                if self.pf['photon_conserving']:
                    self.PhiNdN[absorber] = \
                        10**self.src.tables["logPhi_%s" % absorber](self.logNdN[i],
                        self.logx, t)
                    
                    if (not self.pf['isothermal']) and (self.pf['secondary_ionization'] < 2):
                        self.PsiNdN[absorber] = \
                            10**self.src.tables["logPsi_%s" % absorber](self.logNdN[i],
                            self.logx, t)
            
                if self.pf['secondary_ionization'] > 1:
                    
                    if absorber in self.grid.metals:
                        continue    
                    
                    self.PhiHatN[absorber] = \
                        10**self.src.tables["logPhiHat_%s" % absorber](self.logN_by_cell,
                        self.logx, t)    
                                        
                    if not self.pf['isothermal']:
                        self.PsiHatN[absorber] = \
                            10**self.src.tables["logPsiHat_%s" % absorber](self.logN_by_cell,
                            self.logx, t)  
                                                
                        if self.pf['photon_conserving']:    
                            self.PhiHatNdN[absorber] = \
                                10**self.src.tables["logPhiHat_%s" % absorber](self.logNdN[i],
                                self.logx, t)
                            self.PsiHatNdN[absorber] = \
                                10**self.src.tables["logPsiHat_%s" % absorber](self.logNdN[i],
                                self.logx, t)     
                    
                    for j, donor in enumerate(self.grid.absorbers):
                        
                        suffix = '%s_%s' % (absorber, donor)
                        
                        self.PhiWiggleN[absorber][donor] = \
                            10**self.src.tables["logPhiWiggle_%s" % suffix](self.logN_by_cell,
                                self.logx, t)    
                        
                        self.PsiWiggleN[absorber][donor] = \
                            10**self.src.tables["logPsiWiggle_%s" % suffix](self.logN_by_cell,
                            self.logx, t)
                            
                        if not self.pf['photon_conserving']:
                            continue
                        
                        self.PhiWiggleNdN[absorber][donor] = \
                            10**self.src.tables["logPhiWiggle_%s" % suffix](self.logNdN[j],
                            self.logx, t)
                        self.PsiWiggleNdN[absorber][donor] = \
                            10**self.src.tables["logPsiWiggle_%s" % suffix](self.logNdN[j],
                            self.logx, t)

            # Now, go ahead and calculate the rate coefficients
            for k, absorber in enumerate(self.grid.absorbers):
                self.Gamma[h][...,k] = self.PhotoIonizationRate(absorber)
                self.k_H[h][...,k] = self.PhotoHeatingRate(absorber)

                if absorber in self.grid.metals:
                    continue

                for j, donor in enumerate(self.grid.absorbers):
                    self.gamma[h][...,k,j] = \
                        self.SecondaryIonizationRate(absorber, donor)
                       
            # Compute total optical depth too
            self.tau_tot = 10**self.src.tables["logTau"](self.logN_by_cell)
            
        return self.Gamma, self.gamma, self.k_H, self.Ja
        
    def MultiFreqCoefficients(self, data, absorber):
        """
        Compute all source-dependent rates for given absorber assuming a
        multi-frequency SED.
        """
        
        #k_H = np.zeros_like(self.grid.zeros_grid_x_absorbers)
        #Gamma = np.zeros_like(self.grid.zeros_grid_x_absorbers)
        #gamma = np.zeros_like(self.grid.zeros_grid_x_absorbers2)
        
        k_H = np.zeros(self.grid.dims)
        #Gamma = np.zeros(self.grid.dims)
        gamma = np.zeros_like(self.grid.zeros_grid_x_absorbers)
        
        i = self.grid.absorbers.index(absorber)
        n = self.n[absorber]
        N = self.N[absorber]
               
        # Optical depth up to cells at energy E
        N = np.ones([self.src.Nfreq, self.grid.dims]) * self.N[absorber]
        
        self.tau_r = N * self.sigma[self.h]
        self.tau_tot = np.sum(self.tau_r, axis = 1)
                        
        # Loop over energy groups
        Gamma_E = np.zeros([self.grid.dims, self.src.Nfreq])
        for j, E in enumerate(self.src.E):
            
            if E < self.E_th[absorber]:
                continue    
            
            # Optical depth of cells (at this photon energy)                                                           
            tau_c = self.Nc[absorber] * self.src.sigma[j]
                                                            
            # Photo-ionization by *this* energy group
            Gamma_E[...,j] = \
                self.PhotoIonizationRateMultiFreq(self.src.Qdot[j], n,
                self.tau_r[j], tau_c)            
                          
            # Heating
            if self.grid.isothermal:
                continue
                 
            fheat = self.esec.DepositionFraction(xHII=data['h_2'], 
                E=E, channel='heat')
            
            # Total energy deposition rate per atom i via photo-electrons 
            # due to ionizations by *this* energy group. 
            ee = Gamma_E[...,j] * (E - self.E_th[absorber]) \
               * erg_per_ev
            
            k_H += ee * fheat
                
            if not self.pf['secondary_ionization']:
                continue
                                        
            # Ionizations of species k by photoelectrons from species i
            # Neglect HeII until somebody figures out how that works
            for k, otherabsorber in enumerate(self.grid.absorbers):
            
                # If these photo-electrons don't have enough 
                # energy to ionize species k, continue    
                if (E - self.E_th[absorber]) < \
                    self.E_th[otherabsorber]:
                    continue    
                
                fion = self.esec.DepositionFraction(xHII=data['h_2'], 
                    E=E, channel=absorber)

                # (This k) = i from paper, and (this i) = j from paper
                gamma[...,k,i] += ee * fion \
                    / (self.E_th[otherabsorber] * erg_per_ev)
                                                                           
        # Total photo-ionization tally
        Gamma = np.sum(Gamma_E, axis=1)
        
        return Gamma, gamma, k_H
    
    def PhotoIonizationRateMultiFreq(self, qdot, n, tau_r_E, tau_c):
        """
        Returns photo-ionization rate coefficient for single frequency over
        the entire grid.
        """     
                                        
        q0 = qdot * np.exp(-tau_r_E)             # number of photons entering cell per sec
        dq = q0 * (1. - np.exp(-tau_c))          # number of photons absorbed in cell per sec
        IonizationRate = dq / n / self.grid.Vsh  # ionizations / sec / atom        
                          
        if self.pf['plane_parallel']:
            IonizationRate *= self.pp_corr
        
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
            HeatingRate -= self.E_th[absorber] * erg_per_ev  \
                * self.PhiN[absorber]
            if self.pf['photon_conserving']:
                HeatingRate -= self.PsiNdN[absorber]
                HeatingRate += erg_per_ev \
                    * self.E_th[absorber] \
                    * self.PhiNdN[absorber]
        else:
            HeatingRate = self.PsiHatN[absorber].copy()
            HeatingRate -= self.E_th[absorber] * erg_per_ev  \
                * self.PhiHatN[absorber]
            if self.pf['photon_conserving']:
                HeatingRate -= self.PsiHatNdN[absorber]
                HeatingRate += erg_per_ev \
                    * self.E_th[absorber] \
                    * self.PhiHatNdN[absorber]

        return self.A[absorber] * self.fheat * HeatingRate
            
    def SecondaryIonizationRate(self, absorber, donor):
        """
        Secondary ionization rate which we denote elsewhere as gamma (note little g).
        
            absorber = species being ionized by photo-electron
            donor = species the photo-electron came from
            
        If this routine is called, it means TabulateIntegrals = 1.
        """    
        
        if self.esec.Method < 2:
            IonizationRate = self.PsiN[donor].copy()
            IonizationRate -= self.E_th[donor] \
                * erg_per_ev * self.PhiN[donor]
            if self.pf['photon_conserving']:
                IonizationRate -= self.PsiNdN[donor]
                IonizationRate += self.E_th[donor] \
                    * erg_per_ev * self.PhiNdN[donor]
                            
        else:
            IonizationRate = self.PsiWiggleN[absorber][donor] \
                - self.E_th[donor] \
                * erg_per_ev * self.PhiWiggleN[absorber][donor]
            if self.pf['photon_conserving']:
                IonizationRate -= self.PsiWiggleNdN[absorber][donor]
                IonizationRate += self.E_th[donor] \
                    * erg_per_ev * self.PhiWiggleNdN[absorber][donor]            
                        
        # Normalization (by number densities) will be applied in 
        # chemistry solver    
        return self.A[donor] * self.fion[absorber] * IonizationRate \
                / self.E_th[absorber] / erg_per_ev    
        
        
        