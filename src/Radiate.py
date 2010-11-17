"""
Radiate.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-10-18.

Description: This routine essentially runs the show.  The method 'EvolvePhotons' is the
driver of rt1d, calling our solvers which call all the various physics modules.
     
"""

from RadiationSource import *
from SecondaryElectrons import *
from Interpolate import *
from scipy.integrate import odeint
import numpy as np
import copy

m_e = 9.10938188*10**-28 		# Electron mass - [m_e] = g
m_p = 1.67262158*10**-24		# Proton mass - [m_p] = g
m_n = 1.67492729*10**-24        # Neutron mass - [m_n] = g
k_B = 1.3806503*10**-16			# Boltzmann's constant - [k_B] = erg/K
sigma_T = 6.65*10**-25			# Cross section for Thomson scattering - [sigma_T] = cm^2
h = 6.626068*10**-27 			# Planck's constant - [h] = erg*s
hbar = h / (2 * np.pi) 			# H-bar - [h_bar] = erg*s
c = 29979245800.0 				# Speed of light - [c] = cm/s

m_H = m_p + m_e
m_HeI = 2.0 * (m_p + m_n + m_e)
m_HeII = 2.0 * (m_p + m_n) + m_e

tiny_number = 1e-10

class Radiate:
    def __init__(self, pf, itabs, n_col): 
        self.rs = RadiationSource(pf)
        self.esec = SecondaryElectrons(pf)
        self.pf = pf
        self.itabs = itabs
        self.MultiSpecies = pf["MultiSpecies"]
        self.SolveTemperatureEvolution = pf["SolveTemperatureEvolution"]
        self.InterpolationMethod = pf["InterpolationMethod"]
        self.AdaptiveTimestep = pf["AdaptiveTimestep"]
        self.GridDimensions = pf["GridDimensions"]
        self.InitialRedshift = pf["InitialRedshift"]
        self.LengthUnits = pf["LengthUnits"]
        self.TimeUnits = pf["TimeUnits"]
        self.TimestepSafetyFactor = pf["TimestepSafetyFactor"]
        self.StartRadius = pf["StartRadius"]
        self.StartCell = int(self.StartRadius * self.GridDimensions)
        self.dx = self.LengthUnits / self.GridDimensions
        self.grid = np.arange(self.GridDimensions)
        self.HIColumn = n_col[0]
        self.HeIColumn = n_col[1]
        self.HeIIColumn = n_col[2]
        
        # Always pass three element arrays, the interpolation routines will sort out whether or not they need all three elements.
        if self.MultiSpecies == 0: 
            self.Interpolate = lambda itab, n: Interpolate1D(itab, n_col, n, self.InterpolationMethod)
            self.mu = 1.0 #?
        else: 
            self.Interpolate = lambda itab, n: Interpolate3D(itab, n_col, n, self.InterpolationMethod)
            self.mu = 1.4 #?
        
    def EvolvePhotons(self, data, t, dt):
        """
        This routine calls our solvers and updates 'data'.
        """
        
        newdata = copy.deepcopy(data)
        
        def HIIRateEquation(n_HII_0, t, n_HI, n_e, Gamma_HI, alpha_HII):
            """
            Returns the rate of change of the HII number density.  This is the RHS of Eq. 1 in TZ07.
            
                units: 1 /cm^3 / s
            """
            
            return Gamma_HI * n_HI - alpha_HII * n_e * n_HII_0[0]

        def InternalEnergyRateEquation(T_0, t, nabs, ncol, nion, n_e, x_i, r):
            """
            Returns the rate of change of the gas internal energy.  This is the RHS of Eq. 12 in TZ07, 
            though currently we're missing the Compton heating and Hubble cooling terms.  The conversion
            to temperature units will be done in the body of 'EvolvePhotons'. 
            
                units: erg / cm^3 / s
                notes: did TZ07 forget the Compton cooling term or is it included in the Compton heating
                       term?
            """                        
            
            HeatGain = self.HeatGain(ncol, nabs, x_i, r, t)
            HeatLoss = self.HeatLoss(nabs, nion, n_e, T_0[0])            
                                                                                                                                                                                                                                      
            return HeatGain - HeatLoss
            
        while True:

            HitIonizationLimit = False            

            # Loop over cells radially, solve rate equations, update values in data
            for cell in self.grid:
                
                if (cell * self.dx < (self.StartRadius * self.LengthUnits)): continue
                                             
                # Nice names for all the quantities we need!
                ncol_HI = np.sum(data["HIDensity"][0:cell] * self.dx)
                ncol_HeI = np.sum(data["HeIDensity"][0:cell] * self.dx)
                ncol_HeII = np.sum(data["HeIIDensity"][0:cell] * self.dx)
                ncol = [ncol_HI, ncol_HeI, ncol_HeII]
                n_e = data["ElectronDensity"][cell]
                n_HI = data["HIDensity"][cell]
                n_HII = data["HIIDensity"][cell]
                n_HeI = data["HeIDensity"][cell]
                n_HeII = data["HeIIDensity"][cell]
                n_HeIII = data["HeIIIDensity"][cell]
                x_i = n_HII / (n_HI + n_HII)
                nabs = [n_HI, n_HeI, n_HeII]
                nion = [n_HII, n_HeII, n_HeIII]
                n_B = sum(nabs)
                T = data["Temperature"][cell]
                U = 3. * k_B * T * n_B / self.mu / 2.
                r = self.LengthUnits * cell / self.GridDimensions     
                                                                                                                                                     
                # Some useful quantities for solving the HII rate equation                
                Gamma_HI = self.IonizationRateCoefficientHI(ncol, n_e, x_i, T, r, t)
                alpha_HII = self.RecombinationRateCoefficientHII(T)
                
                # Compute timestep based on ionization timescale in closest cell to source
                if self.AdaptiveTimestep and cell == self.StartCell: 
                    dt = (1. / Gamma_HI) * self.TimestepSafetyFactor
                
                newHII = odeint(HIIRateEquation, [n_HII, 0], [0, dt], \
                    args = (n_HI, n_e, Gamma_HI, alpha_HII,), mxstep = 1000)[1][0] 
                                         
                newHI = (n_HI + n_HII) - newHII  
                                                
                # Next, solve the heat equation  
                if self.SolveTemperatureEvolution:                     
                    newU = odeint(InternalEnergyRateEquation, [U, 0], [0, dt], \
                        args = (nabs, ncol, nion, n_e, x_i, r,), mxstep = 1000)[1][0]   
                        
                    newT = newU * 2. * self.mu / 3. / k_B / n_B 
                else: newT = T
                    
                # Solve helium rate equations (not yet implemented)
                newHeII = data["HeIIDensity"][cell]
                newHeIII = data["HeIIIDensity"][cell]
                newHeI = (n_HeI + n_HeII + n_HeIII) - newHeII - newHeIII
                                                                                                                                                                                                                        
                # Update quantities in 'data'.     
                newdata["HIDensity"][cell] = newHI                                                                                                                                          
                newdata["HIIDensity"][cell] = newHII
                newdata["HeIDensity"][cell] = newHeI
                newdata["HeIIDensity"][cell] = newHeII
                newdata["HeIIIDensity"][cell] = newHeIII
                newdata["ElectronDensity"][cell] = newHII + newdata["HeIIDensity"][cell] + 2.0 * newdata["HeIIIDensity"][cell]
                newdata["Temperature"][cell] = newT
          
            if HitIonizationLimit is False: break  
                                
        return newdata, dt
        
    def RecombinationRateCoefficientHII(self, T):
        """
        Returns recombination rate coefficient, using approximation of Zaroubi et al. 2007.
        
            units: cm^3 / s
        """
        return 2.6e-13 * (T / 1.e4)**-0.85
        
    def IonizationRateCoefficientHI(self, ncol, n_e, x_i, T, r, t):
        """
        Returns ionization rate coefficient for HI, which we denote elsewhere as Gamma_HI.  Includes photo, collisional, 
        and secondary ionizations from fast electrons.
        
            units: 1 / s
        """            
        PhotoIonizationTerm = self.rs.BolometricLuminosity(t) * self.Interpolate(self.itabs["PhotoIonizationRateIntegralHI"], ncol) / 4. / np.pi / r**2      
        CollisionalIonizationTerm = self.rs.BolometricLuminosity(t) * n_e * 5.85e-11 * np.sqrt(T) * (1. + np.sqrt(T / 1.e5))**-1. * np.exp(-1.578e5 / T) / 4. / np.pi / r**2
        SecondaryIonizationTerm = self.esec.DepositionFraction(0.0, x_i, channel = 1) * self.rs.BolometricLuminosity(t) * self.Interpolate(self.itabs["SecondaryIonizationRateIntegralHI"], ncol) / 4. / np.pi / r**2
                                                                                                                                                                                                                                                                          
        return PhotoIonizationTerm + CollisionalIonizationTerm + SecondaryIonizationTerm
        
    def HeatGain(self, ncol, nabs, x_i, r, t):
        """
        Returns the total heating rate at radius r and time t.  These are all the terms in Eq. 12 of TZ07 on
        the RHS that are positive.
        
            units: erg / s / cm^3
        """
                         
        heat = 0           
        for i, integral in enumerate(["ElectronHeatingIntegralHI", "ElectronHeatingIntegralHeI", "ElectronHeatingIntegralHeII"]): 
            try: heat += nabs[i] * self.Interpolate(self.itabs[integral], ncol)
            except KeyError: pass
                  
        heat *= self.esec.DepositionFraction(0.0, x_i, channel = 0) * self.rs.BolometricLuminosity(t) / 4.0 / np.pi / r**2 
                                                                        
        return heat
    
    def HeatLoss(self, nabs, nion, n_e, T):
        """
        Returns the total cooling rate for a cell of temperature T and with species densities given in 'nabs', 'nion', and 'n_e'. 
        This quantity is the sum of all terms on the RHS of Eq. 12 in TZ07 that are negative (except for the Hubble cooling term), 
        though we do not apply the minus sign until later, in 'ThermalRateEquation'.
        
            units: erg / s / cm^3
        """
            
        T_cmb = 2.725 * (1. + self.InitialRedshift)    
        cool = 0
        
        # Cooling by collisional ionization
        for i, n in enumerate(nabs):
            cool += n * self.CollisionalIonizationCoolingCoefficient(T, i)
                
        # Cooling by collisional excitation
        for i, n in enumerate(nabs):
            cool += n * self.CollisionalExcitationCoolingCoefficient(T, nabs, nion, i)
        
        # Cooling by recombinations
        for i, n in enumerate(nion):
            cool += n * self.RecombinationCoolingCoefficient(T, i)
                        
        # Cooling by dielectronic recombination
        cool += nion[2] * self.DielectricRecombinationCoolingCoefficient(T)
        
        # Compton cooling
        cool += 4. * k_B * (T - T_cmb) * (np.pi**2 / 15.) * (k_B * T_cmb / hbar / c)**3 * (k_B * T_cmb / m_e / c**2) * sigma_T * c
        
        # Cooling by free-free emission
        cool += sum(nion) * 1.42e-27 * 1.1 * np.sqrt(T)     # Check on Gaunt factor
                
        cool *= n_e
        
        return cool
        
    def HubbleCooling(self):
        pass
        
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
        if species == 2: return 5.54e-17 * T**-0.397 * (1. + np.sqrt(T / 1e5))**-1. * np.exp(-4.73e5 / T)    
        
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
        
    
        
        
        
        