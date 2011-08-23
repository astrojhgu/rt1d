"""
Radiate.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-10-18.

Description: This routine essentially runs the show.  The method 'EvolvePhotons' is the
driver of rt1d, calling our solvers which call all the various physics modules.
     
"""

import numpy as np
import copy, scipy
from rt1d.mods.RadiationSource import RadiationSource
from rt1d.mods.SecondaryElectrons import SecondaryElectrons
from rt1d.mods.Interpolate import Interpolate
from rt1d.mods.Cosmology import Cosmology
from rt1d.mods.SolveRateEquations import SolveRateEquations
from progressbar import *

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    print "Module mpi4py not found.  No worries, we'll just run in serial."
    rank = 0
    size = 1

m_e = 9.10938188*10**-28 		# Electron mass - [m_e] = g
m_p = 1.67262158*10**-24		# Proton mass - [m_p] = g
m_n = 1.67492729*10**-24        # Neutron mass - [m_n] = g
k_B = 1.3806503*10**-16			# Boltzmann's constant - [k_B] = erg/K
sigma_T = 6.65*10**-25			# Cross section for Thomson scattering - [sigma_T] = cm^2
h = 6.626068*10**-27 			# Planck's constant - [h] = erg*s
hbar = h / (2 * np.pi) 			# H-bar - [h_bar] = erg*s
c = 29979245800.0 			    # Speed of light - [c] = cm/s

m_H = m_p + m_e
m_HeI = 2.0 * (m_p + m_n + m_e)
m_HeII = 2.0 * (m_p + m_n) + m_e

min_density = 1e-12

# Widget for progressbar.
widget = ["Ray Casting: ", Percentage(), ' ', Bar(marker = RotatingMarker()), ' ', ETA(), ' ']

class Radiate:
    def __init__(self, pf, data, itabs, n_col): 
        self.rs = RadiationSource(pf)
        self.esec = SecondaryElectrons(pf)
        self.cosmo = Cosmology(pf)
        self.pf = pf
        self.itabs = itabs
        
        self.ParallelizationMethod = pf["ParallelizationMethod"]
        
        self.MaxHIIChange = pf["MaxHIIChange"]
        self.MaxHeIIChange = pf["MaxHeIIChange"]
        self.HIIRestrictedTimestep = pf["HIIRestrictedTimestep"]
        self.HeIIRestrictedTimestep = pf["HeIIRestrictedTimestep"]
        
        self.MultiSpecies = pf["MultiSpecies"]
        self.InfiniteSpeedOfLight = pf["InfiniteSpeedOfLight"]
        self.Isothermal = pf["Isothermal"]
        self.ComptonCooling = pf["ComptonCooling"]
        self.CollisionalIonization = pf["CollisionalIonization"]
        self.CollisionalExcitation = pf["CollisionalExcitation"]
        self.SecondaryIonization = pf["SecondaryIonization"]
        self.InitialTemperature = pf["InitialTemperature"]
        self.PlaneParallelField = pf["PlaneParallelField"]
        
        self.InterpolationMethod = pf["InterpolationMethod"]
        self.AdaptiveTimestep = pf["ODEAdaptiveStep"]
        self.CosmologicalExpansion = pf["CosmologicalExpansion"]
        self.InitialHIIFraction = pf["InitialHIIFraction"]
        self.GridDimensions = pf["GridDimensions"]
        self.InitialRedshift = pf["InitialRedshift"]
        self.LengthUnits = pf["LengthUnits"]
        self.TimeUnits = pf["TimeUnits"]
        self.StopTime = pf["StopTime"] * self.TimeUnits
        self.StartRadius = pf["StartRadius"]
        self.StartCell = int(self.StartRadius * self.GridDimensions)
        self.InitialHydrogenDensity = (data["HIDensity"][0] + data["HIIDensity"][0]) / (1. + self.InitialRedshift)**3
        self.InitialHeliumDensity = (data["HeIDensity"][0] + data["HeIIDensity"][0] + data["HeIIIDensity"][0]) / (1. + self.InitialRedshift)**3
        self.grid = np.arange(self.GridDimensions)
        self.r = self.LengthUnits * self.grid / self.GridDimensions  
        self.dx = self.LengthUnits / self.GridDimensions
        self.HIColumn = n_col[0]
        self.HeIColumn = n_col[1]
        self.HeIIColumn = n_col[2]
        
        self.OnePhotonPackagePerCell = pf["OnePhotonPackagePerCell"]
        self.RadiativeTransferCourantCondition = pf["RadiativeTransferCourantCondition"]        
        self.LightCrossingTimeRestrictedStep = pf["LightCrossingTimeRestrictedStep"]
        self.AdaptiveStep = pf["ODEAdaptiveStep"]
        self.MaxStep = pf["ODEMaxStep"] * self.TimeUnits
        self.MinStep = pf["ODEMinStep"] * self.TimeUnits
        self.atol = pf["ODEatol"]
        self.rtol = pf["ODErtol"]
        
        self.ProgressBar = pf["ProgressBar"]                                                                        
        
        guesses = [data["HIDensity"][0] + data["HIIDensity"][0], 
                   data["HeIDensity"][0] + data["HeIIDensity"][0] + data["HeIIIDensity"][0], 
                   data["HeIDensity"][0] + data["HeIIDensity"][0] + data["HeIIIDensity"][0], 
                   3 * self.InitialTemperature * k_B * (self.InitialHydrogenDensity + self.InitialHeliumDensity) / 2.]
                
        self.solver = SolveRateEquations(pf, guesses = guesses, stepper = self.AdaptiveStep, hmin = self.MinStep, hmax = self.MaxStep, \
            rtol = self.rtol, atol = self.atol, Dfun = None, maxiter = pf["ODEmaxiter"])
                                
        self.Interpolate = Interpolate(self.pf, n_col, self.itabs)                        
                                
        self.Y = 0.2477 * self.MultiSpecies
        self.X = 1. - self.Y

    def qdot(self, q, t, *args):
        """
        This function returns the right-hand side of our ODE's.

        q = [n_HII, n_HeII, n_HeIII, E] - our four coupled equations. q = generalized quantity I guess.

        for q[0, 1, 2]: units: 1 /cm^3 / s
        for q[3]: units: erg / cm^3 / s

        args = ([r, z, mu, n_H, n_He, ncol],)       
        
        Should probably calculate most of this stuff once, so it needn't be on each iteration of the solver.
        
        """

        # Extra arguments
        r = args[0][0]                         
        z = args[0][1]
        mu = args[0][2]
        n_H = args[0][3]
        n_He = args[0][4]
        ncol = args[0][5]
        Lbol = args[0][6]
                        
        # Derived quantities
        n_HI = n_H - q[0]
        n_HII = q[0]
        x_HII = n_HII / n_H
        n_HeI = n_He - (q[1] + q[2]) 
        n_HeII = q[1]
        n_HeIII = q[2]
        n_e = n_HII + n_HeII + 2.0 * n_HeIII
        nion = [n_HII, n_HeII, n_HeIII]
        nabs = [n_H, n_HeI, n_HeII]
        n_B = n_H + n_He + n_e

        E = q[3]        
        if self.Isothermal: T = self.InitialTemperature
        else: T = E * 2. * mu / 3. / k_B / n_B
        
        # First, solve for rate coefficients
        alpha_HII = 2.6e-13 * (T / 1.e4)**-0.85    
        Gamma_HI = self.IonizationRateCoefficientHI(ncol, n_e, n_HI, n_HeI, x_HII, T, r, Lbol)        
                                                                                             
        if self.MultiSpecies > 0: 
            Gamma_HeI = self.IonizationRateCoefficientHeI(ncol, n_HI, n_HeI, x_HII, T, r, Lbol)
            Gamma_HeII = self.IonizationRateCoefficientHeII(ncol, x_HII, r, Lbol)
            Beta_HeI = 2.38e-11 * np.sqrt(T) * (1. + np.sqrt(T / 1.e5))**-1. * np.exp(-2.853e5 / T) * self.CollisionalIonization
            Beta_HeII = 5.68e-12 * np.sqrt(T) * (1. + np.sqrt(T / 1.e5))**-1. * np.exp(-6.315e5 / T) * self.CollisionalIonization
            alpha_HeII = 9.94e-11 * T**-0.48                                                            ## WHICH ALPHA_HEIII IS RIGHT FOR US?
            alpha_HeIII = 3.36e-10 * T**-0.5 * (T / 1e3)**-0.2 * (1. + (T / 4.e6)**0.7)**-1.
            if T < 2.2e4: alpha_HeIII *= (1.11 - 0.044 * np.log(T))
            else: alpha_HeIII *= (1.43 - 0.076 * np.log(T))
            xi_HeII = 1.9e-3 * T**-1.5 * np.exp(-4.7e5 / T) * (1. + 0.3 * np.exp(-9.4e4 / T))
        else: Gamma_HeI = Gamma_HeII = Beta_HeI = Beta_HeII = alpha_HeII = alpha_HeIII = alpha_HeIII = xi_HeII = 0
                                                                
        # Always solve hydrogen rate equation
        newHII = Gamma_HI * n_HI - alpha_HII * n_e * q[0]
       
        # Only solve helium rate equations if self.MultiSpeces = 1
        if self.MultiSpecies:
            newHeII = Gamma_HeI * n_HeI - Beta_HeI * n_e * n_HeI + Beta_HeII * n_e * q[1] - \
                      alpha_HeII * n_e * q[1] + alpha_HeIII * n_e * n_HeIII - xi_HeII * n_e * q[1]    
            newHeIII = Gamma_HeII * n_HeII - Beta_HeII * n_e * n_HeII + alpha_HeIII * n_e * q[2]
        else:
            newHeII = q[1]
            newHeIII = q[2]

        # Only solve internal energy equation if we're not doing an isothermal calculation
        if self.Isothermal: 
            newE = q[3]
        else:
            newE = self.HeatGain(ncol, nabs, x_HII, r, Lbol) - \
                self.HeatLoss(nabs, nion, n_e, n_B, q[3] * 2. * mu / 3. / k_B / n_B, z, mu)
                                                                                                
        return np.array([newHII, newHeII, newHeIII, newE])

    def EvolvePhotons(self, data, t, dt, h):
        """
        This routine calls our solvers and updates 'data'.
        """
        
        newdata = {}
        for key in data.keys(): newdata[key] = copy.deepcopy(data[key])
        z = self.cosmo.TimeToRedshiftConverter(0., t, self.InitialRedshift)

        # Nice names for ionized fractions
        x_HI_arr = data["HIDensity"] / (data["HIDensity"] + data["HIIDensity"])
        x_HII_arr = data["HIIDensity"] / (data["HIDensity"] + data["HIIDensity"])
        
        if self.MultiSpecies:
            x_HeI_arr = data["HeIDensity"] / (data["HeIDensity"] + data["HeIIDensity"] + data["HeIIIDensity"])
            x_HeII_arr = data["HeIIDensity"] / (data["HeIDensity"] + data["HeIIDensity"] + data["HeIIIDensity"])
            x_HeIII_arr = data["HeIIIDensity"] / (data["HeIDensity"] + data["HeIIDensity"] + data["HeIIIDensity"])
        
        # This is not a good idea in general, but in this case they'll never be touched again.
        else: x_HeI_arr = x_HeII_arr = x_HeIII_arr = np.zeros_like(x_HI_arr)
                                                        
        # If we're in an expanding universe, dilute densities by (1 + z)^3    
        if self.CosmologicalExpansion: 
            data["HIDensity"] = x_HI * self.InitialHydrogenDensity * (1. + z)**3
            data["HIIDensity"] = x_HII * self.InitialHydrogenDensity * (1. + z)**3
            data["HeIDensity"] = x_HeI * self.InitialHeliumDensity * (1. + z)**3
            data["HeIIDensity"] = x_HeII * self.InitialHeliumDensity * (1. + z)**3
            data["HeIIIDensity"] = x_HeIII * self.InitialHeliumDensity * (1. + z)**3    
            data["ElectronDensity"] = data["HIIDensity"] + data["HeIIDensity"] + 2. * data["HeIIIDensity"]

        # Compute column densities
        ncol_HI = np.cumsum(data["HIDensity"]) * self.dx
        ncol_HeI = np.cumsum(data["HeIDensity"]) * self.dx
        ncol_HeII = np.cumsum(data["HeIIDensity"]) * self.dx
                        
        # Print status, and update progress bar
        if rank == 0: print "rt1d: {0} < t < {1}".format(round(t / self.TimeUnits, 8), round((t + dt) / self.TimeUnits, 8))            
        if rank == 0 and self.ProgressBar: pbar = ProgressBar(widgets = widget, maxval = self.grid[-1]).start()

        if self.HIIRestrictedTimestep: dtphot = np.ones_like(self.grid) * 1e50
        
        # Deal with c < infinity
        if not self.InfiniteSpeedOfLight: 
            
            # Photon packages going from oldest to youngest - will have to create it on first timestep
            try: packs = list(data['PhotonPackages']) 
            except KeyError: packs = []            
            
            if packs and self.OnePhotonPackagePerCell:            
                t_packs = zip(*packs)[0]    # Birth times for all packages
                r_packs = (t - t_packs) * c
                
                # If there are already photon packages in the first cell, don't make another
                if np.any(r_packs[r_packs < (self.StartRadius * self.LengthUnits)]): pass
                else:
                    # Launch new photon package - [t_birth, ncol_HI_0, ncol_HeI_0, ncol_HeII_0]                
                    packs.append(np.array([t, 0., 0., 0.]))
            else: packs.append(np.array([t, 0., 0., 0.]))
                                                                 
        # If accreting black hole, luminosity will change with time.
        Lbol = self.rs.BolometricLuminosity(t)

        # Loop over cells radially, solve rate equations, update values in data -> newdata
        for cell in self.grid:
            
            if (cell % size != rank) and (self.ParallelizationMethod == 1): continue
                        
            # If within our buffer zone (where we don't solve rate equations), continue
            if cell < self.StartCell: continue
                                                
            if rank == 0 and self.ProgressBar: pbar.update(cell)
            
            # Read in densities for this cell
            n_e = data["ElectronDensity"][cell]
            n_HI = data["HIDensity"][cell]
            n_HII = data["HIIDensity"][cell]
            n_HeI = data["HeIDensity"][cell]
            n_HeII = data["HeIIDensity"][cell]
            n_HeIII = data["HeIIIDensity"][cell] 

            # Read in ionized fractions for this cell
            x_HI = x_HI_arr[cell]
            x_HII = x_HII_arr[cell]
            x_HeI = x_HeI_arr[cell]
            x_HeII = x_HeII_arr[cell]
            x_HeIII = x_HeIII_arr[cell]

            # Compute mean molecular weight for this cell
            mu = 1. / (self.X * (1. + x_HII) + self.Y * (1. + x_HeII + x_HeIII) / 4.)
                                    
            # For convenience     
            ncol = [ncol_HI[cell], ncol_HeI[cell], ncol_HeII[cell]]
            nabs = [n_HI, n_HeI, n_HeII]
            nion = [n_HII, n_HeII, n_HeIII]
            n_H = n_HI + n_HII
            n_He = n_HeI + n_HeII + n_HeIII
            n_B = n_H + n_He + n_e              # STILL DONT UNDERSTAND THIS
                                    
            # Compute internal energy for this cell
            T = data["Temperature"][cell]
            E = 3. * k_B * T * n_B / mu / 2.

            # Compute radius
            r = self.r[cell]

            ######################################
            ######## Solve Rate Equations ########
            ######################################
            
            if self.InfiniteSpeedOfLight:
                tarr, qnew, h = self.solver.integrate(self.qdot, (n_HII, n_HeII, n_HeIII, E), t, t + dt, None, h, \
                    r, z, mu, n_H, n_He, ncol, Lbol)
            
            # Things are tougher if c != inf        
            else:
                values = (n_HII, n_HeII, n_HeIII, E)
                qnew = [[0, n_HII], [0, n_HeII], [0, n_HeIII], [0, E]]
                                                                                                                                               
                # Loop over photon packages and solve rate equations
                for j, pack in enumerate(packs):       
                    t_birth = pack[0]
                    r_pack = (t - t_birth) * c                      # Position of package before evolving photons
                    r_max = (t + dt - t_birth) * c                  # Furthest this package will get this timestep
                    
                    cell_pack = r_pack * self.GridDimensions / self.LengthUnits
                    cell_pack_max = int(r_max * self.GridDimensions / self.LengthUnits)
                                                                                                                  
                    # If this photon package is already past the current cell, do nothing
                    if cell_pack > cell: continue                   
                                        
                    # If this photon package won't reach the current cell in the next dt, none of them will.  Proceed.
                    if cell > cell_pack_max: break
                                                                                            
                    # Column density (and thus tau) between source and where this package was at time t
                    ncol_pack = copy.copy(pack[1:])
                                        
                    # Add optical depth of the medium this package has traversed only in the last dt
                    ncol_pack[0] += np.sum(newdata['HIDensity'][cell_pack:cell]) * self.dx
                    ncol_pack[1] += np.sum(newdata['HeIDensity'][cell_pack:cell]) * self.dx
                    ncol_pack[2] += np.sum(newdata['HeIIDensity'][cell_pack:cell]) * self.dx
                                                                                                                       
                    # Luminosity of object at time package was launched
                    Lbol_new = self.rs.BolometricLuminosity(t_birth)

                    subdt = min((r_max - r) / c, self.dx - (r_pack - r) / c, self.dx / c)
                                                                            
                    #print subdt, (r_max - r) / c, self.dx - (r_pack - r) / c, self.dx / c         
                                                                                                                
                    tarr, qnew, h = self.solver.integrate(self.qdot, values, t, t + subdt, None, h, \
                        r, z, mu, n_H, n_He, ncol_pack, Lbol_new)
                    
                    newdata['HIDensity'][cell] = n_H - qnew[0][-1]
                    newdata['HIIDensity'][cell] = n_H - newdata['HIDensity'][cell]
                    newdata['HeIDensity'][cell] = n_He - qnew[1][-1] - qnew[2][-1]
                    newdata['HeIIDensity'][cell] = n_He - qnew[2][-1] - newdata['HeIDensity'][cell]
                    newdata['HeIIIDensity'][cell] = n_He - newdata['HeIIDensity'][cell] - newdata['HeIDensity'][cell]
                    newdata["ElectronDensity"][cell] = (n_H - newdata['HIDensity'][cell]) + newdata['HeIIDensity'][cell] + 2.0 * newdata['HeIIIDensity'][cell]
                    
                    x_HII = newdata['HIIDensity'][cell] / n_H
                    if self.MultiSpecies:
                        x_HeII = newdata['HeIIDensity'][cell] / n_He
                        x_HeIII = newdata['HeIIIDensity'][cell] / n_He
                    else: x_HeII = x_HeIII = 0
                           
                    if not self.Isothermal:                    
                        mu = 1. / (self.X * (1. + x_HII) + self.Y * (1. + x_HeII + x_HeIII) / 4.)
                        n_B = newdata['HIDensity'][cell] + newdata['HIIDensity'][cell] + newdata["ElectronDensity"][cell] # NOT GOOD FOR MULTISPECIES
                        newdata["Temperature"][cell] = qnew[3][-1] * 2. * mu / 3. / k_B / n_B
                    else: qnew[3][-1] = E
                                        
                    values = (qnew[0][-1], qnew[1][-1], qnew[2][-1], qnew[3][-1])

                    # Not currently worrying about tolerance error stuff here - see below
            
                                                                                    
            # Unpack results of coupled equations - remember, these are lists and we only need the last entry 
            newHII, newHeII, newHeIII, newE = qnew
                        
            # Convert from internal energy back to temperature
            newT = newE[-1] * 2. * mu / 3. / k_B / n_B

            # Possible that newHII > n_H and within tolerances, hence the min statements
            if newHII[-1] > n_H: newHI = min_density
            else: newHI = n_H - newHII[-1]      
            
            # Same problem could arise with helium - favor accuracy of HeII over HeIII
            newHeI = n_He - newHeII[-1] - newHeIII[-1]
            if newHeI < 0: 
                newHeI = min_density
                newerHeII = n_He - newHeI - newHeIII[-1]
                newerHeIII = n_He - newHeI - newerHeII
            else:
                newHeI = n_He - newHeII[-1] - newHeIII[-1]
                newerHeII = n_He - newHeI - newHeIII[-1]
                newerHeIII = n_He - newHeI - newerHeII
                
            # Update quantities in 'data' -> 'newdata'     
            newdata["HIDensity"][cell] = newHI                                                                                                                        
            newdata["HIIDensity"][cell] = n_H - newHI
            newdata["HeIDensity"][cell] = newHeI
            newdata["HeIIDensity"][cell] = newerHeII
            newdata["HeIIIDensity"][cell] = newerHeIII
            newdata["ElectronDensity"][cell] = (n_H - newHI) + newerHeII + 2.0 * newerHeIII
            newdata["Temperature"][cell] = newT
            
            ######################################
            ################ DONE ################     
            ######################################
                        
            # Calculate global timstep based on change in hydrogen neutral fraction for next iteration
            if self.HIIRestrictedTimestep:  
                newxHII = newHII[-1] / n_H                
                
                newncol = [np.cumsum(newdata["HIDensity"])[cell] * self.dx, np.cumsum(newdata["HeIDensity"])[cell] * self.dx,
                           np.cumsum(newdata["HeIIDensity"])[cell] * self.dx]
                
                Gamma = self.IonizationRateCoefficientHI(newncol, newdata["ElectronDensity"][cell], newHI, newHeI, newxHII, newT, r, Lbol)        
                alpha = 2.6e-13 * (newT / 1.e4)**-0.85  
                
                # Shapiro et al. 2004
                dtphot[cell] = self.MaxHIIChange * newHI / np.abs(newHI * Gamma - newHII[-1] * newdata["ElectronDensity"][cell] * alpha)  

                # Calculate global timstep based on change in helium neutral fraction for next iteration
                if self.MultiSpecies and self.HeIIRestrictedTimestep:
                    newxHeII = newerHeII / n_He
                    
                    Beta = 2.38e-11 * np.sqrt(newT) * (1. + np.sqrt(newT / 1.e5))**-1. * np.exp(-2.853e5 / newT) * self.CollisionalIonization
                    Gamma = self.IonizationRateCoefficientHeI(newncol, newHI, newHeI, newxHII, newT, r, Lbol)                    
                    alpha = 9.94e-11 * newT**-0.48   
                    
                    # Analogous to Shapiro et al. 2004 but for helium
                    dtphot[cell] = min(dtphot[cell], self.MaxHeIIChange * newHeI / \
                        np.abs(newHeI * (Gamma + newdata["ElectronDensity"][cell] * Beta) - newerHeII* newdata["ElectronDensity"][cell] * alpha)) 
                    
                # So timestep can be written out later
                newdata["dtPhoton"][cell] = dtphot[cell]                      
                          
        if (size > 0) and (self.ParallelizationMethod == 1):
            for key in newdata.keys(): newdata[key] = MPI.COMM_WORLD.allreduce(newdata[key], newdata[key])
        
        if self.HIIRestrictedTimestep: newdt = min(np.min(dtphot[self.StartCell:]), 2 * dt)
        else: newdt = dt
        
        if self.LightCrossingTimeRestrictedStep: newdt = min(newdt, self.RadiativeTransferCourantCondition * self.LengthUnits / self.GridDimensions / 29979245800.0)
        
        if rank == 0 and self.ProgressBar: pbar.finish()
        
        # Update photon packages        
        if not self.InfiniteSpeedOfLight: newdata['PhotonPackages'] = self.UpdatePhotonPackages(packs, t + dt, newdata)  # t + newdt?      
                
        return newdata, h, newdt         
        
    def UpdatePhotonPackages(self, packs, t_next, data):
        """
        Get rid of old photon packages, and update column densities and corresponding time.
        """    
        
        ncol_HI = np.cumsum(data["HIDensity"]) * self.dx
        ncol_HeI = np.cumsum(data["HeIDensity"]) * self.dx
        ncol_HeII = np.cumsum(data["HeIIDensity"]) * self.dx
        
        for i, pack in enumerate(packs):
            t_birth = pack[0]
            r_pack = (t_next - t_birth) * c               
            cell_pack = int(r_pack * self.GridDimensions / self.LengthUnits)
            
            if cell_pack > (self.GridDimensions / self.LengthUnits): continue
            
            packs[i][1] = ncol_HI[cell_pack]
            packs[i][2] = ncol_HeI[cell_pack]
            packs[i][3] = ncol_HeII[cell_pack]                        
                
        to_eliminate = []
        for i, pack in enumerate(packs):
            if (t_next - pack[0]) > (self.LengthUnits / c): to_eliminate.append(i)
            if pack[0] > self.HIColumn[-1]: 
                to_eliminate.append(i)
            
        to_eliminate.reverse()    
        for element in to_eliminate: packs.pop(element)
        
        return np.array(packs)
        
    def IonizationRateCoefficientHI(self, ncol, n_e, n_HI, n_HeI, x_HII, T, r, Lbol):
        """
        Returns ionization rate coefficient for HI, which we denote elsewhere as Gamma_HI.  Includes photo, collisional, 
        and secondary ionizations from fast electrons.
        
            units: 1 / s
        """     
           
        # Photo-Ionization       
        IonizationRate = Lbol * \
                         self.Interpolate.interp(ncol, "PhotoIonizationRate{0}".format(0)) \
                         / 4. / np.pi / r**2   
                         
        if self.PlaneParallelField: IonizationRate *= 4. * np.pi * r**2                      
        
        # Collisional Ionization
        if self.CollisionalIonization:
            IonizationRate += n_e * 5.85e-11 * np.sqrt(T) * (1. + np.sqrt(T / 1.e5))**-1. * np.exp(-1.578e5 / T)
        
        if self.SecondaryIonization:
            IonizationRate += Lbol * \
                              self.esec.DepositionFraction(0.0, x_HII, channel = 1) * \
                              self.Interpolate.interp(ncol, "SecondaryIonizationRateHI{0}".format(0)) \
                              / 4. / np.pi / r**2     
                              
            if self.PlaneParallelField: IonizationRate *= 4. * np.pi * r**2                                                                     
                        
            if self.MultiSpecies > 0:
                IonizationRate += Lbol * (n_HeI / n_HI) * \
                                 self.esec.DepositionFraction(0.0, x_HII, channel = 1) * \
                                 self.Interpolate.interp(ncol, "SecondaryIonizationRateHI{0}".format(1)) \
                                 / 4. / np.pi / r**2
                                 
                if self.PlaneParallelField: IonizationRate *= 4. * np.pi * r**2                                       
                
        return IonizationRate
        
    def IonizationRateCoefficientHeI(self, ncol, n_HI, n_HeI, x_HII, T, r, Lbol):
        """
        Returns ionization rate coefficient for HeI, which we denote elsewhere as Gamma_HeI.  Includes photo 
        and secondary ionizations from fast electrons.  Unlike the hydrogen case, the collisional ionizations
        are included in the rate equation itself instead of a coefficient.
        
            units: 1 / s
        """                
        
        IonizationRate = Lbol * \
                         self.Interpolate.interp(ncol, "PhotoIonizationRate{0}".format(1)) \
                         / 4. / np.pi / r**2 
                         
        if self.PlaneParallelField: IonizationRate *= 4. * np.pi * r**2                                       
        
        if self.SecondaryIonization:
            IonizationRate += Lbol * \
                              self.esec.DepositionFraction(0.0, x_HII, channel = 2) * \
                              self.Interpolate.interp(ncol, "SecondaryIonizationRateHeI{0}".format(1)) \
                              / 4. / np.pi / r**2
            
            IonizationRate += (n_HI / n_HeI) * Lbol * \
                              self.esec.DepositionFraction(0.0, x_HII, channel = 2) * \
                              self.Interpolate.interp(ncol, "SecondaryIonizationRateHeI{0}".format(1)) \
                              / 4. / np.pi / r**2
                              
            if self.PlaneParallelField: IonizationRate *= 4. * np.pi * r**2                                        
        
        return IonizationRate
        
    def IonizationRateCoefficientHeII(self, ncol, x_HII, r, Lbol):
        """
        Returns ionization rate coefficient for HeII, which we denote elsewhere as Gamma_HeII.  Includes photo 
        and secondary ionizations from fast electrons.  Unlike the hydrogen case, the collisional ionizations
        are included in the rate equation itself instead of a coefficient.  Note: TZ07 do not include secondary
        helium II ionizations, but I am going to.
        
            units: 1 / s
        """       
        
        IonizationRate = Lbol * self.Interpolate.interp(ncol, "PhotoIonizationRate{0}".format(2)) / 4. / np.pi / r**2 
        
        if self.PlaneParallelField: IonizationRate *= 4. * np.pi * r**2                      
        
        if self.SecondaryIonization > 1:
            IonizationRate += Lbol * self.esec.DepositionFraction(0.0, x_HII, channel = 3) * \
                self.Interpolate.interp(ncol, "SecondaryIonizationRate{0}".format(2)) / 4. / np.pi / r**2
                
            if self.PlaneParallelField: IonizationRate *= 4. * np.pi * r**2                          
        
        return IonizationRate
        
    def HeatGain(self, ncol, nabs, x_HII, r, Lbol):
        """
        Returns the total heating rate at radius r and time t.  These are all the terms in Eq. 12 of TZ07 on
        the RHS that are positive.
        
            units: erg / s / cm^3
        """
                                 
        heat = nabs[0] * self.Interpolate.interp(ncol, "ElectronHeatingRate{0}".format(0))

        if self.MultiSpecies > 0:
            heat += nabs[1] * self.Interpolate.interp(ncol, "ElectronHeatingRate{0}".format(1))
            heat += nabs[2] * self.Interpolate.interp(ncol, "ElectronHeatingRate{0}".format(2))
                              
        heat *= self.esec.DepositionFraction(0.0, x_HII, channel = 0) * Lbol / 4.0 / np.pi / r**2 
        
        if self.PlaneParallelField: heat *= 4. * np.pi * r**2                      
                                                                                                                                                                                                                         
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
                
        # Cooling by collisional excitation
        if self.CollisionalExcitation:
            for i, n in enumerate(nabs):
                cool += n * self.CollisionalExcitationCoolingCoefficient(T, nabs, nion, i)
        
        # Cooling by recombinations
        for i, n in enumerate(nion):
            cool += n * self.RecombinationCoolingCoefficient(T, i)
                        
        # Cooling by dielectronic recombination
        cool += nion[2] * self.DielectricRecombinationCoolingCoefficient(T)
        
        # Compton cooling - from FK96
        if self.ComptonCooling:
            cool += 4. * k_B * (T - T_cmb) * (np.pi**2 / 15.) * (k_B * T_cmb / hbar / c)**3 * (k_B * T_cmb / m_e / c**2) * sigma_T * c
        
        ## Cooling by free-free emission
        #cool += (nion[0] + nion[1] + 4. * nion[2]) * 1.42e-27 * 1.1 * np.sqrt(T) # Check on Gaunt factor        
                
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
        