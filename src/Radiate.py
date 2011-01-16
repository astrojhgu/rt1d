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
from Cosmology import *
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
    def __init__(self, pf, data, itabs, n_col): 
        self.rs = RadiationSource(pf)
        self.esec = SecondaryElectrons(pf)
        self.cosmo = Cosmology(pf)
        self.pf = pf
        self.itabs = itabs
        self.MultiSpecies = pf["MultiSpecies"]
        self.SolveTemperatureEvolution = pf["SolveTemperatureEvolution"]
        self.InterpolationMethod = pf["InterpolationMethod"]
        self.InitialTimestep = pf["InitialTimestep"] * pf["TimeUnits"]
        self.AdaptiveTimestep = pf["AdaptiveTimestep"]
        self.MaxHIIFraction = pf["MaxHIIFraction"]
        self.CosmologicalExpansion = pf["CosmologicalExpansion"]
        self.InitialHIIFraction = pf["InitialHIIFraction"]
        self.GridDimensions = pf["GridDimensions"]
        self.InitialRedshift = pf["InitialRedshift"]
        self.LengthUnits = pf["LengthUnits"]
        self.TimeUnits = pf["TimeUnits"]
        self.TimestepSafetyFactor = pf["TimestepSafetyFactor"]
        self.StartRadius = pf["StartRadius"]
        self.StartCell = int(self.StartRadius * self.GridDimensions)
        self.InitialHydrogenDensity = (data["HIDensity"][0] + data["HIIDensity"][0]) / (1. + self.InitialRedshift)**3
        self.InitialHeliumDensity = (data["HeIDensity"][0] + data["HeIIDensity"][0] + data["HeIIIDensity"][0]) / (1. + self.InitialRedshift)**3
        self.dx = self.LengthUnits / self.GridDimensions
        self.grid = np.arange(self.GridDimensions)
        self.HIColumn = n_col[0]
        self.HeIColumn = n_col[1]
        self.HeIIColumn = n_col[2]
                                
        # Always pass three element arrays, the interpolation routines will sort out whether or not they need all three elements.
        if self.MultiSpecies == 0: 
            self.Interpolate = lambda itab, n: Interpolate1D(itab, n_col, n, self.InterpolationMethod)
            self.Y = 0.
            self.X = 1.
        else: 
            self.Interpolate = lambda itab, n: Interpolate3D(itab, n_col, n, self.InterpolationMethod)
            self.Y = 0.2477
            self.X = 1. - Y
        
    def EvolvePhotons(self, data, t, dt):
        """
        This routine calls our solvers and updates 'data'.
        """
        
        newdata = copy.deepcopy(data)
        z = self.cosmo.TimeToRedshiftConverter(0., t, self.InitialRedshift)
        maxdt = 0.0
        
        def HIIRateEquation(n_HII_0, t, n_HI, n_e, Gamma_HI, alpha_HII):
            """
            Returns the rate of change of the HII number density.  This is the RHS of Eq. 1 in TZ07.
            
                units: 1 /cm^3 / s
            """
                                                                        
            return Gamma_HI * n_HI - alpha_HII * n_e * n_HII_0[0]
            
        def HIIRateEqJacobian(n_HII_0, t, n_HI, n_e, Gamma_HI, alpha_HII):
            """
            Gradient of the HII rate equation, this somehow helps the ODE solver.
            """

            return [[0, t], [- alpha_HII * n_e, 0]]

        def HeIIRateEquation(n_HeII_0, t, n_HeI, n_e, Gamma_HeI, Beta_HeI, Beta_HeII, alpha_HeII, alpha_HeIII, xi_HeII):
            """
            Returns the rate of change of the HeII number density.  This is the RHS of Eq. 2 in TZ07.
            
                units: 1 /cm^3 / s
            """
            
            return Gamma_HeI * n_HeI - Beta_HeI * n_e * n_HeI + Beta_HeII * n_e * n_HeII_0[0] - \
                alpha_HeII * n_e * n_HeII_0[0] + alpha_HeIII * n_e * n_HeIII - xi_HeII * n_e * n_HeII_0[0]
        
        def HeIIRateEqJacobian(n_HeII_0, t, n_HeI, n_e, Gamma_HeI, Beta_HeI, Beta_HeII, alpha_HeII, alpha_HeIII, xi_HeII):
            """
            Gradient of the HeII rate equation, this somehow helps the ODE solver.
            """

            return [[0, t], [Beta_HeII * n_e - alpha_HeII * n_e - xi_HeII * n_e, 0]] 
         
                
        def HeIIIRateEquation(n_HeIII_0, t, n_HeII, n_e, Gamma_HeII, Beta_HeII, alpha_HeIII):
            """
            Returns the rate of change of the HeIII number density.  This is the RHS of Eq. 3 in TZ07.
            
                units: 1 /cm^3 / s
            """
                        
            return Gamma_HeII * n_HeII - Beta_HeII * n_e * n_HeII + alpha_HeIII * n_e * n_HeIII_0[0]

        def HeIIIRateEqJacobian(n_HeIII_0, t, n_HeII, n_e, Gamma_HeII, Beta_HeII, alpha_HeIII):
            """
            Gradient of the HeIII rate equation, this somehow helps the ODE solver.
            """

            return [[0, t], [alpha_HeIII * n_e, 0]]

        def InternalEnergyRateEquation(T_0, t, nabs, ncol, nion, n_e, n_B, x_i, z, mu, r):
            """
            Returns the rate of change of the gas internal energy.  This is the RHS of Eq. 12 in TZ07, 
            though currently we're missing the Compton heating and Hubble cooling terms.  The conversion
            to temperature units will be done in the body of 'EvolvePhotons'. 
            
                units: erg / cm^3 / s
                notes: did TZ07 forget the Compton cooling term or is it included in the Compton heating
                       term?
            """                        
            
            HeatGain = self.HeatGain(ncol, nabs, x_i, r, t)
            HeatLoss = self.HeatLoss(nabs, nion, n_e, n_B, T_0[0], z, mu)            
                                                                                                                                                                                                                                      
            return HeatGain - HeatLoss

        # The meat of the matter, with some array-optimization.
            
        # Nice names for all the quantities we need!
        x_HI_arr = data["HIDensity"] / (data["HIDensity"] + data["HIIDensity"])
        x_HII_arr = data["HIIDensity"] / (data["HIDensity"] + data["HIIDensity"])

        if self.MultiSpecies:
            x_HeI_arr = data["HeIDensity"] / (data["HeIDensity"] + data["HeIIDensity"] + data["HeIIIDensity"])
            x_HeII_arr = data["HeIIDensity"] / (data["HeIDensity"] + data["HeIIDensity"] + data["HeIIIDensity"])
            x_HeIII_arr = data["HeIIIDensity"] / (data["HeIDensity"] + data["HeIIDensity"] + data["HeIIIDensity"])
        
        else: x_HeI_arr = x_HeII_arr = x_HeIII_arr = np.zeros_like(x_HI_arr)
                                                        
        # If we're in an expanding universe, dilute densities by (1 + z)^3    
        if self.CosmologicalExpansion: 
            data["HIDensity"] = x_HI * self.InitialHydrogenDensity * (1. + z)**3
            data["HIIDensity"] = x_HII * self.InitialHydrogenDensity * (1. + z)**3
            data["HeIDensity"] = x_HeI * self.InitialHeliumDensity * (1. + z)**3
            data["HeIIDensity"] = x_HeII * self.InitialHeliumDensity * (1. + z)**3
            data["HeIIIDensity"] = x_HeIII * self.InitialHeliumDensity * (1. + z)**3    
            data["ElectronDensity"] = data["HIIDensity"] + data["HeIIDensity"] + 2. * data["HeIIIDensity"]

        ncol_HI = np.cumsum(data["HIDensity"]) * self.dx
        ncol_HeI = np.cumsum(data["HeIDensity"]) * self.dx
        ncol_HeII = np.cumsum(data["HeIIDensity"]) * self.dx

        # Loop over cells radially, solve rate equations, update values in data
        for cell in self.grid:
        
            if cell < self.StartCell: continue
            
            n_e = data["ElectronDensity"][cell]
            n_HI = data["HIDensity"][cell]
            n_HII = data["HIIDensity"][cell]
            n_HeI = data["HeIDensity"][cell]
            n_HeII = data["HeIIDensity"][cell]
            n_HeIII = data["HeIIIDensity"][cell] 
            
            x_HI = x_HI_arr[cell]
            x_HII = x_HII_arr[cell]
            mu = 1. / (self.X * (1. + x_HII))
            if self.MultiSpecies:
                x_HeI = x_HeI_arr[cell]
                x_HeII = x_HeII_arr[cell]
                x_HeIII = x_HeIII_arr[cell]
                mu = 1. / (self.X * (1. + x_HII) + self.Y * (1. + x_HeII + x_HeIII) / 4.)
            
            ncol = [ncol_HI[cell], ncol_HeI[cell], ncol_HeII[cell]]
            nabs = [n_HI, n_HeI, n_HeII]
            nion = [n_HII, n_HeII, n_HeIII]
            n_H = n_HI + n_HII
            n_He = n_HeI + n_HeII + n_HeIII
            n_B = n_H + n_He + n_e
            T = data["Temperature"][cell]
            U = 3. * k_B * T * n_B / mu / 2.
            r = self.LengthUnits * cell / self.GridDimensions  
                                                                                                                                                                                                                                           
            # Some useful quantities for solving the HII rate equation                
            Gamma_HI = self.IonizationRateCoefficientHI(ncol, n_e, n_HI, n_HeI, x_HII, T, r, t)
            alpha_HII = 2.6e-13 * (T / 1.e4)**-0.85
            
            # Some useful quantities for solving the HeII and HeIII rate equations
            if self.MultiSpecies > 0: 
                Gamma_HeI = self.IonizationRateCoefficientHeI(ncol, n_HI, n_HeI, x_HII, r, t)
                Gamma_HeII = self.IonizationRateCoefficientHeII(ncol, x_HII, r, t)
                Beta_HeI = 2.38e-11 * np.sqrt(T) * (1. + np.sqrt(T / 1.e5))**-1. * np.exp(-2.853e5 / T)
                Beta_HeII = 5.68e-12 * np.sqrt(T) * (1. + np.sqrt(T / 1.e5))**-1. * np.exp(-6.315e5 / T)
                alpha_HeII = 9.94e-11 * T**-0.48
                alpha_HeIII = 3.36e-10 * T**-0.5 * (T / 1e3)**-0.2 * (1. + (T / 4.e6)**0.7)**-1.
                if T < 2.2e4: alpha_HeIII *= (1.11 - 0.044 * np.log(T))
                else: alpha_HeIII *= (1.43 - 0.076 * np.log(T))
                xi_HeII = 1.9e-3 * T**-1.5 * np.exp(-4.7e5 / T) * (1. + 0.3 * np.exp(-9.4e4 / T))
                                                                    
            # New adaptive timestepping - this info will be used on the next timestep 
            if self.AdaptiveTimestep > 0:
                newmaxdt = n_H / (Gamma_HI * n_HI - alpha_HII * n_e * n_HII)
                if not np.isfinite(newmaxdt): newmaxdt = 0.0
                
                if self.MultiSpecies > 0:
                    maxdt_He = n_He / \
                        (Gamma_HeI * n_HeI - Beta_HeI * n_e * n_HeI + Beta_HeII * n_e * n_HeII - \
                        alpha_HeII * n_e * n_HeII + alpha_HeIII * n_e * n_HeIII - xi_HeII * n_e * n_HeII)
                
                    if not np.isfinite(maxdt_He): maxdt_He = 0.0
                    
                    newmaxdt = max(newmaxdt, maxdt_He)

                maxdt = self.TimestepSafetyFactor * max(newmaxdt, maxdt)
                  
            # Solve the HII rate equation                                                                                                                                  
            newHII = odeint(HIIRateEquation, [n_HII, 0], [0, dt], Dfun = HIIRateEqJacobian, \
                args = (n_HI, n_e, Gamma_HI, alpha_HII,), mxstep = 10000)[1][0]
                
            # Hack                                                                                            
            if newHII > n_H: 
                newHII = self.MaxHIIFraction * n_H
                    
            newHI = n_H - newHII
                
            # Solve the helium rate equations
            if self.MultiSpecies > 0:
                
                # Solve the HeII rate equation   
                newHeII = odeint(HeIIRateEquation, [n_HeII, 0], [0, dt], Dfun = HeIIRateEqJacobian, \
                    args = (n_HeI, n_e, Gamma_HeI, Beta_HeI, Beta_HeII, alpha_HeII, alpha_HeIII, xi_HeII,), \
                    mxstep = 10000)[1][0]
                                        
                # Solve the HeIII rate equation
                newHeIII = odeint(HeIIIRateEquation, [n_HeIII, 0], [0, dt], Dfun = HeIIIRateEqJacobian, \
                    args = (n_HeII, n_e, Gamma_HeII, Beta_HeII, alpha_HeIII,), \
                    mxstep = 10000)[1][0]
                                                                    
                # Hack                                                                                            
                if newHeII + newHeIII > n_He or not np.isfinite(newHeII + newHeIII): 
                    newHeII = self.MaxHIIFraction * n_He
                    newHeIII = 0.0
                
                newHeI = n_He - newHeII + newHeIII
            
            else:
                newHeI = data["HeIDensity"][cell]
                newHeII = data["HeIIDensity"][cell]
                newHeIII = data["HeIIIDensity"][cell] 
                
            # Next, solve the heat equation  
            if self.SolveTemperatureEvolution:                     
                newU = odeint(InternalEnergyRateEquation, [U, 0], [0, dt], \
                    args = (nabs, ncol, nion, n_e, n_B, x_HII, z, mu, r,), mxstep = 10000)[1][0]   
                    
                newT = newU * 2. * mu / 3. / k_B / n_B 
            else: newT = T            
                                                                                                                                                                                                                                    
            # Update quantities in 'data'.     
            newdata["HIDensity"][cell] = newHI                                                                                                                                          
            newdata["HIIDensity"][cell] = newHII
            newdata["HeIDensity"][cell] = newHeI
            newdata["HeIIDensity"][cell] = newHeII
            newdata["HeIIIDensity"][cell] = newHeIII
            newdata["ElectronDensity"][cell] = newHII + newHeII + 2.0 * newHeIII
            newdata["Temperature"][cell] = newT
           
        if self.AdaptiveTimestep > 0: nextdt = min(self.InitialTimestep, maxdt) 
        else: nextdt = dt            
                                        
        return newdata, nextdt
        
    def IonizationRateCoefficientHI(self, ncol, n_e, n_HI, n_HeI, x_HII, T, r, t):
        """
        Returns ionization rate coefficient for HI, which we denote elsewhere as Gamma_HI.  Includes photo, collisional, 
        and secondary ionizations from fast electrons.
        
            units: 1 / s
        """     
               
        PhotoIonizationTerm = self.rs.BolometricLuminosity(t) * self.Interpolate(self.itabs["PhotoIonizationRateIntegralHI"], ncol) / 4. / np.pi / r**2      
        CollisionalIonizationTerm = self.rs.BolometricLuminosity(t) * n_e * 5.85e-11 * np.sqrt(T) * (1. + np.sqrt(T / 1.e5))**-1. * np.exp(-1.578e5 / T) / 4. / np.pi / r**2
        SecondaryIonizationTerm_H = self.esec.DepositionFraction(0.0, x_HII, channel = 1) * self.rs.BolometricLuminosity(t) * self.Interpolate(self.itabs["SecondaryIonizationRateIntegralHI_HI"], ncol) / 4. / np.pi / r**2
        if self.MultiSpecies > 0: SecondaryIonizationTerm_He = self.esec.DepositionFraction(0.0, x_HII, channel = 1) * (n_HeI / n_HI) * self.rs.BolometricLuminosity(t) * self.Interpolate(self.itabs["SecondaryIonizationRateIntegralHI_HeI"], ncol) / 4. / np.pi / r**2                        
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
        return PhotoIonizationTerm + CollisionalIonizationTerm + SecondaryIonizationTerm_H #+ SecondaryIonizationTerm_He
        
    def IonizationRateCoefficientHeI(self, ncol, n_HI, n_HeI, x_HII, r, t):
        """
        Returns ionization rate coefficient for HeI, which we denote elsewhere as Gamma_HeI.  Includes photo 
        and secondary ionizations from fast electrons.  Unlike the hydrogen case, the collisional ionizations
        are included in the rate equation itself instead of a coefficient.
        
            units: 1 / s
        """                
        
        PhotoIonizationTerm = self.rs.BolometricLuminosity(t) * self.Interpolate(self.itabs["PhotoIonizationRateIntegralHeI"], ncol) / 4. / np.pi / r**2 
        SecondaryIonizationTerm_H = self.esec.DepositionFraction(0.0, x_HII, channel = 2) * self.rs.BolometricLuminosity(t) * self.Interpolate(self.itabs["SecondaryIonizationRateIntegralHeI_HI"], ncol) / 4. / np.pi / r**2
        if self.MultiSpecies > 0: SecondaryIonizationTerm_He = self.esec.DepositionFraction(0.0, x_HII, channel = 2) * (n_HI / n_HeI) * self.rs.BolometricLuminosity(t) * self.Interpolate(self.itabs["SecondaryIonizationRateIntegralHeI_HeI"], ncol) / 4. / np.pi / r**2
                
        return PhotoIonizationTerm + SecondaryIonizationTerm_H + SecondaryIonizationTerm_He
        
    def IonizationRateCoefficientHeII(self, ncol, x_HII, r, t):
        """
        Returns ionization rate coefficient for HeII, which we denote elsewhere as Gamma_HeII.  Includes photo 
        and secondary ionizations from fast electrons.  Unlike the hydrogen case, the collisional ionizations
        are included in the rate equation itself instead of a coefficient.  Note: TZ07 do not include secondary
        helium II ionizations, but I am going to.
        
            units: 1 / s
        """       
        
        PhotoIonizationTerm = self.rs.BolometricLuminosity(t) * self.Interpolate(self.itabs["PhotoIonizationRateIntegralHeII"], ncol) / 4. / np.pi / r**2 
        #SecondaryIonizationTerm = self.esec.DepositionFraction(0.0, x_HII, channel = 3) * self.rs.BolometricLuminosity(t) * self.Interpolate(self.itabs["SecondaryIonizationRateIntegralHeII"], ncol) / 4. / np.pi / r**2
        
        return PhotoIonizationTerm #+ SecondaryIonizationTerm
        
    def HeatGain(self, ncol, nabs, x_HII, r, t):
        """
        Returns the total heating rate at radius r and time t.  These are all the terms in Eq. 12 of TZ07 on
        the RHS that are positive.
        
            units: erg / s / cm^3
        """
                         
        heat = 0           
        for i, integral in enumerate(["ElectronHeatingIntegralHI", "ElectronHeatingIntegralHeI", "ElectronHeatingIntegralHeII"]): 
            try: heat += nabs[i] * self.Interpolate(self.itabs[integral], ncol)
            except KeyError: pass
                  
        heat *= self.esec.DepositionFraction(0.0, x_HII, channel = 0) * self.rs.BolometricLuminosity(t) / 4.0 / np.pi / r**2 
                                                                                              
        return heat
    
    def HeatLoss(self, nabs, nion, n_e, n_B, T, z, mu):
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
        
        # Hubble cooling
        if self.CosmologicalExpansion: cool += 2. * self.cosmo.HubbleParameter(z) * (k_B * T * n_B / mu)
        
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
        
    
        
        
        
        