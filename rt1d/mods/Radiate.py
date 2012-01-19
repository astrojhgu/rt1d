"""
Radiate.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-10-18.

Description: This routine essentially runs the show.  The method 'EvolvePhotons' is the
driver of rt1d, calling our solvers which call all the various physics modules.
     
"""

import copy, math
import numpy as np
from rt1d.mods.ComputeCrossSections import PhotoIonizationCrossSection
from rt1d.mods.RadiationSource import RadiationSource
from rt1d.mods.SecondaryElectrons import SecondaryElectrons
from rt1d.mods.Interpolate import Interpolate
from rt1d.mods.Cosmology import Cosmology
from rt1d.mods.ComputeRateCoefficients import RateCoefficients
from rt1d.mods.SolveRateEquations import SolveRateEquations
from rt1d.mods.ControlSimulation import ControlSimulation
from Integrate import simpson as integrate

try:
    from progressbar import *
    pb = True
except ImportError:
    print "Module progressbar not found."
    pb = False

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

E_th = [13.6, 24.6, 54.4]

neglible_column = 1

# Widget for progressbar.
if pb: 
    widget = ["Ray Casting: ", Percentage(), ' ', Bar(marker = RotatingMarker()), ' ', ETA(), ' ']

class Radiate:
    def __init__(self, pf, data, itabs, n_col): 
        self.pf = pf
        self.rs = RadiationSource(pf)
        self.cosmo = Cosmology(pf)
        self.coeff = RateCoefficients(pf, itabs = itabs, n_col = n_col)
        self.control = ControlSimulation(pf)
        
        self.PhotonConserving = pf['PhotonConserving']
        
        self.ParallelizationMethod = pf["ParallelizationMethod"]
        self.debug = pf["Debug"]
        
        self.MaxHIIChange = pf["MaxHIIChange"]
        self.MaxHeIIChange = pf["MaxHeIIChange"]
        self.MaxHeIIIChange = pf["MaxHeIIIChange"]
        self.HIIRestrictedTimestep = pf["HIIRestrictedTimestep"]
        self.HeIIRestrictedTimestep = pf["HeIIRestrictedTimestep"]
        self.HeIIIRestrictedTimestep = pf["HeIIIRestrictedTimestep"]
        
        self.MultiSpecies = pf["MultiSpecies"]
        self.InfiniteSpeedOfLight = pf["InfiniteSpeedOfLight"]
        self.Isothermal = pf["Isothermal"]
        self.ComptonCooling = pf["ComptonCooling"]
        self.CollisionalIonization = pf["CollisionalIonization"]
        self.CollisionalExcitation = pf["CollisionalExcitation"]
        self.SecondaryIonization = pf["SecondaryIonization"]
        self.FreeFreeEmission = pf["FreeFreeEmission"]
        self.InitialTemperature = pf["InitialTemperature"]
        self.PlaneParallelField = pf["PlaneParallelField"]
        
        self.InterpolationMethod = pf["InterpolationMethod"]
        self.AdaptiveTimestep = pf["ODEAdaptiveStep"]
        self.CosmologicalExpansion = pf["CosmologicalExpansion"]
        self.InitialHIIFraction = pf["InitialHIIFraction"]
        self.GridDimensions = int(pf["GridDimensions"])
        self.LengthUnits = pf["LengthUnits"]
        self.TimeUnits = pf["TimeUnits"]
        self.StopTime = pf["StopTime"] * self.TimeUnits
        self.StartRadius = pf["StartRadius"]
        self.R0 = self.StartRadius * self.LengthUnits
        self.InitialRedshift = pf["InitialRedshift"]
        self.InitialHydrogenDensity = (data["HIDensity"][0] + data["HIIDensity"][0]) / (1. + self.InitialRedshift)**3
        self.InitialHeliumDensity = (data["HeIDensity"][0] + data["HeIIDensity"][0] + data["HeIIIDensity"][0]) / (1. + self.InitialRedshift)**3
        self.HIColumn = n_col[0]
        self.HeIColumn = n_col[1]
        self.HeIIColumn = n_col[2]
        
        self.LogGrid = pf['LogarithmicGrid']
        self.grid = np.arange(self.GridDimensions)
            
        # Deal with log-grid
        if self.LogGrid:
            self.r = np.logspace(np.log10(self.StartRadius * self.LengthUnits), \
                np.log10(self.LengthUnits), self.GridDimensions)
            r_tmp = np.concatenate([[0], self.r])
            self.dx = np.diff(r_tmp)    
        else:
            self.dx = self.LengthUnits / self.GridDimensions
            rmin = max(self.dx, self.StartRadius * self.LengthUnits)
            self.r = np.linspace(rmin, self.LengthUnits, self.GridDimensions)
                    
        self.CellCrossingTime = self.dx / c
        self.OnePhotonPackagePerCell = pf["OnePhotonPackagePerCell"]
        self.LightCrossingTimeRestrictedStep = pf["LightCrossingTimeRestrictedStep"]
        self.AdaptiveStep = pf["ODEAdaptiveStep"]
        self.MaxStep = pf["ODEMaxStep"] * self.TimeUnits
        self.MinStep = pf["ODEMinStep"] * self.TimeUnits
        self.atol = pf["ODEatol"]
        self.rtol = pf["ODErtol"]
        
        self.ProgressBar = pf["ProgressBar"] and pb                                                                 
        
        self.TotalHydrogen = data["HIDensity"][-1] + data["HIIDensity"][-1]
        self.TotalHelium = data["HeIDensity"][-1] + data["HeIIDensity"][-1] + data["HeIIIDensity"][-1]
        guesses = [0.5 * self.TotalHydrogen, 
                   0.5 * self.TotalHelium, 
                   0.1 * self.TotalHelium, 
                   3 * self.InitialTemperature * k_B * (self.TotalHydrogen + self.TotalHelium) / 2.]
                
        # Initialize solver        
        self.solver = SolveRateEquations(pf, guesses = guesses, stepper = self.AdaptiveStep, hmin = self.MinStep, hmax = self.MaxStep, \
            rtol = self.rtol, atol = self.atol, Dfun = None, maxiter = pf["ODEmaxiter"])
        
        # Initialize interpolation routines                        
        #self.Interpolate = Interpolate(self.pf, n_col, self.itabs)                        
        
        # Initialize helium abundance                        
        self.Y = 0.2477 * self.MultiSpecies
        self.X = 1. - self.Y

    def EvolvePhotons(self, data, t, dt, h, lb):
        """
        This routine calls our solvers and updates 'data'.
        """
        
        # Do this so an MPI all-reduce doesn't add stuff together
        if self.ParallelizationMethod == 1 and size > 1:
            solve_arr = np.arange(self.GridDimensions)
            proc_mask = np.zeros_like(solve_arr)
            condition = (solve_arr >= lb[rank]) & (solve_arr < lb[rank + 1])
            proc_mask[condition] = 1
            solve_arr = solve_arr[proc_mask == 1]  
                          
        # Set up newdata dictionary                                
        newdata = {}
        for key in data.keys(): 
            newdata[key] = copy.deepcopy(data[key])
            
            if self.ParallelizationMethod == 1 and size > 1:
                newdata[key][proc_mask == 0] = 0        
             
        z = self.cosmo.TimeToRedshiftConverter(0., t, self.InitialRedshift)
        
        # Nice names for ionized fractions
        n_H_arr = data["HIDensity"] + data["HIIDensity"]
        x_HI_arr = data["HIDensity"] / n_H_arr
        x_HII_arr = data["HIIDensity"] / n_H_arr
        
        if self.MultiSpecies:
            n_He_arr = data["HeIDensity"] + data["HeIIDensity"] + data["HeIIIDensity"]
            x_HeI_arr = data["HeIDensity"] / n_He_arr
            x_HeII_arr = data["HeIIDensity"] / n_He_arr
            x_HeIII_arr = data["HeIIIDensity"] / n_He_arr
        
        # This is not a good idea in general, but in this case they'll never be touched again.
        else: 
            n_He_arr = x_HeI_arr = x_HeII_arr = x_HeIII_arr = np.zeros_like(x_HI_arr)
                                                        
        # If we're in an expanding universe, dilute densities by (1 + z)**3    
        if self.CosmologicalExpansion: 
            data["HIDensity"] = x_HI * self.InitialHydrogenDensity * (1. + z)**3
            data["HIIDensity"] = x_HII * self.InitialHydrogenDensity * (1. + z)**3
            data["HeIDensity"] = x_HeI * self.InitialHeliumDensity * (1. + z)**3
            data["HeIIDensity"] = x_HeII * self.InitialHeliumDensity * (1. + z)**3
            data["HeIIIDensity"] = x_HeIII * self.InitialHeliumDensity * (1. + z)**3    
            data["ElectronDensity"] = data["HIIDensity"] + data["HeIIDensity"] + 2. * data["HeIIIDensity"]

        # Compute column densities
        ncol_HI = np.cumsum(data["HIDensity"] * self.dx) 
        ncol_HeI = np.cumsum(data["HeIDensity"] * self.dx) 
        ncol_HeII = np.cumsum(data["HeIIDensity"] * self.dx) 
        ncol_HI[0] = ncol_HeI[0] = ncol_HeII[0] = neglible_column
        
        # Convenience arrays for column, absorber, and ion densities, plus some others
        ncol_all = np.transpose([ncol_HI, ncol_HeI, ncol_HeII])
        nabs_all = np.transpose([data["HIDensity"], data["HeIDensity"], data["HeIIDensity"]])
        nion_all = np.transpose([data["HIIDensity"], data["HeIIDensity"], data["HeIIIDensity"]])
        mu_all = 1. / (self.X * (1. + x_HII_arr) + self.Y * (1. + x_HeII_arr + x_HeIII_arr) / 4.)
        nB_all = n_H_arr + n_He_arr + data["ElectronDensity"]
        q_all = np.transpose([data["HIIDensity"], data["HeIIDensity"], data["HeIIIDensity"], 3. * k_B * data["Temperature"] * nB_all / mu_all / 2.])

        # Print status, and update progress bar
        if rank == 0: print "rt1d: {0} < t < {1}".format(round(t / self.TimeUnits, 8), round((t + dt) / self.TimeUnits, 8))            
        if rank == 0 and self.ProgressBar: pbar = ProgressBar(widgets = widget, maxval = self.grid[-1]).start()
        
        # If accreting black hole, luminosity will change with time.
        Lbol = self.rs.BolometricLuminosity(t)
        
        # Set up 'packs' structure for c < infinity runs
        if not self.InfiniteSpeedOfLight: 
            
            # PhotonPackage guide: pack = [EmissionTime, EmissionTimeInterval, ncolHI, ncolHeI, ncolHeII, Energy]
            
            # Photon packages going from oldest to youngest - will have to create it on first timestep
            try: packs = list(data['PhotonPackages']) 
            except KeyError: packs = []            
            
            if packs and self.OnePhotonPackagePerCell:
                t_packs = np.array(zip(*packs)[0])    # Birth times for all packages
                r_packs = (t - t_packs) * c
                
                # If there are already photon packages in the first cell, add energy in would-be packet to preexisting one
                if r_packs[-1] < (self.StartRadius * self.LengthUnits):
                    packs[-1][-1] += Lbol * dt
                else:
                    # Launch new photon package - [t_birth, ncol_HI_0, ncol_HeI_0, ncol_HeII_0]                
                    packs.append(np.array([t, dt, 0., 0., 0., Lbol * dt]))
            else: packs.append(np.array([t, dt, 0., 0., 0., Lbol * dt]))
        
        # Initialize timestep array
        if self.HIIRestrictedTimestep: 
            if self.InfiniteSpeedOfLight:
                dtphot = 1. * np.zeros_like(self.grid)    
            else:
                dtphot = 1e50 * np.ones_like(self.grid)
                
        else:
            dtphot = dt                
        
        ###
        ## SOLVE: c -> inf
        ###        
        if self.InfiniteSpeedOfLight:
            
            # Loop over cells radially, solve rate equations, update values in data -> newdata
            for cell in self.grid:
                            
                # If this cell belongs to another processor, continue
                if self.ParallelizationMethod == 1 and size > 1:
                    if cell not in solve_arr: continue
                        
                # Update progressbar        
                if rank == 0 and self.ProgressBar: 
                    pbar.update(cell * size)
                                                                
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
                                        
                # Convenience arrays for column, absorber, and ion densities plus a few others
                ncol = ncol_all[cell]
                nabs = nabs_all[cell]
                nion = nion_all[cell]
                n_H = n_HI + n_HII
                n_He = n_HeI + n_HeII + n_HeIII                
                n_B = n_H + n_He + n_e
                
                # Compute mean molecular weight for this cell
                mu = 1. / (self.X * (1. + x_HII) + self.Y * (1. + x_HeII + x_HeIII) / 4.)
                                                        
                # Compute internal energy for this cell
                T = data["Temperature"][cell]
                E = 3. * k_B * T * n_B / mu / 2.
                        
                # Read radius
                r = self.r[cell]
                
                # Retrieve indices used for 3D interpolation
                indices = None
                if self.MultiSpecies > 0 and not self.PhotonConserving: 
                    indices = self.coeff.Interpolate.GetIndices3D(ncol)
                
                # Retrieve coefficients and what not.
                args = [nabs, n_H, n_He, n_e]
                args.extend(self.coeff.ConstructArgs(args, indices, Lbol, r, ncol, T))
                
                # Unpack so we have everything by name
                nabs, n_H, n_He, n_e, Gamma, gamma, Beta, alpha, k_H, zeta, eta, psi = args
                                            
                ######################################
                ######## Solve Rate Equations ########
                ######################################                
                                
                tarr, qnew, h = self.solver.integrate(self.RateEquations, q_all[cell], t, t + dt, 
                    None, h, *args)
                                                                                                         
                # Unpack results of coupled equations
                newHII, newHeII, newHeIII, newE = qnew 
                                                
                # Convert from internal energy back to temperature
                newT = newE * 2. * mu / 3. / k_B / n_B   
                
                # Calculate neutral fractions
                newHI = n_H - newHII
                newHeI = n_He - newHeII - newHeIII
                
                # Update quantities in 'data' -> 'newdata'                
                newdata["HIDensity"][cell] = newHI                                                                                            
                newdata["HIIDensity"][cell] = n_H - newHI
                newdata["HeIDensity"][cell] = newHeI
                newdata["HeIIDensity"][cell] = newHeII
                newdata["HeIIIDensity"][cell] = newHeIII
                newdata["ElectronDensity"][cell] = (n_H - newHI) + newHeIII + 2.0 * newHeIII
                newdata["Temperature"][cell] = newT        
                                                                              
                # Adjust timestep based on maximum allowed neutral fraction change                              
                if self.HIIRestrictedTimestep:
                    dtphot[cell] = self.control.ComputePhotonTimestep(self.coeff.tau, 
                        Gamma, gamma, Beta, alpha, 
                        nabs, nion, ncol, n_H, n_He, n_e)                            
                                
                ######################################
                ################ DONE ################
                ######################################
                
        ###
        ## SOLVE: c = finite
        ###   
        else:                
       
            # Loop over photon packages, updating values in cells: data -> newdata
            for j, pack in enumerate(packs):
                t_birth = pack[0]
                r_pack = (t - t_birth) * c        # Position of package before evolving photons
                r_max = r_pack + dt * c           # Furthest this package will get this timestep
                                                                                 
                # Cells we need to know about - not necessarily integer
                cell_pack = r_pack * self.GridDimensions / self.LengthUnits
                cell_pack_max = r_max * self.GridDimensions / self.LengthUnits
                              
                Lbol = pack[-1] / pack[1]          
                                                
                # Advance this photon package as far as it will go on this global timestep  
                while cell_pack < cell_pack_max:
                                                            
                    # What cell are we in
                    cell = int(round(cell_pack))
                    
                    if cell >= self.GridDimensions: break
                    
                    # Compute dc (like dx but in fractional cell units)
                    if cell_pack % 1 == 0: dc = min(cell_pack_max - cell_pack, 1)
                    else: dc = min(math.ceil(cell_pack) - cell_pack, cell_pack_max - cell_pack)        
                                                         
                    # We really need to evolve this cell until the next photon package arrives, which
                    # is probably longer than a cell crossing time unless the global dt is vv small.
                    if (len(packs) > 1) and ((j + 1) < len(packs)): subdt = min(dt, packs[j + 1][0] - pack[0])
                    else: subdt = dt
                                                                                                                                                                                                                                                                                                                                              
                    r = cell_pack * self.LengthUnits / self.GridDimensions
                    
                    if r < (self.StartRadius / self.LengthUnits): 
                        cell_pack += dc
                        continue
                    
                    n_e = newdata["ElectronDensity"][cell]
                    n_HI = newdata["HIDensity"][cell]
                    n_HII = newdata["HIIDensity"][cell]
                    n_HeI = newdata["HeIDensity"][cell]
                    n_HeII = newdata["HeIIDensity"][cell]
                    n_HeIII = newdata["HeIIIDensity"][cell] 
                    
                    # Read in ionized fractions for this cell
                    x_HI = x_HI_arr[cell]
                    x_HII = x_HII_arr[cell]
                    x_HeI = x_HeI_arr[cell]
                    x_HeII = x_HeII_arr[cell]
                    x_HeIII = x_HeIII_arr[cell]
                    
                    # Compute mean molecular weight for this cell
                    mu = 1. / (self.X * (1. + x_HII) + self.Y * (1. + x_HeII + x_HeIII) / 4.)
                                            
                    # For convenience     
                    nabs = [n_HI, n_HeI, n_HeII]
                    nion = [n_HII, n_HeII, n_HeIII]
                    n_H = n_HI + n_HII
                    n_He = n_HeI + n_HeII + n_HeIII
                    n_B = n_H + n_He + n_e
                                            
                    # Compute internal energy for this cell
                    T = newdata["Temperature"][cell]
                    E = 3. * k_B * T * n_B / mu / 2.
                    
                    # Crossing time will depend on cell for logarithmic grids
                    if self.LogGrid:
                        dct = self.CellCrossingTime[cell]
                    else:
                        dct = self.CellCrossingTime
                    
                    # Add columns of this cell
                    packs[j][2] += newdata['HIDensity'][cell] * dc * dct * c  
                    packs[j][3] += newdata['HeIDensity'][cell] * dc * dct * c 
                    packs[j][4] += newdata['HeIIDensity'][cell] * dc * dct * c
                    
                    ######################################
                    ######## Solve Rate Equations ########
                    ######################################
                    
                    # Retrieve indices used for 3D interpolation
                    indices = None
                    if self.MultiSpecies > 0: 
                        indices = self.coeff.Interpolate.GetIndices3D(ncol)
                    
                    qnew = np.array([n_HII, n_HeII, n_HeIII, E])

                    tarr, qnew, h = self.solver.integrate(self.qdot, qnew, t, t + subdt, None, h, \
                        r, z, mu, n_H, n_He, packs[j][2:5], Lbol, indices)
                                        
                    # Unpack results of coupled equations - Remember: these are lists and we only need the last entry 
                    newHII, newHeII, newHeIII, newE = qnew

                    # Convert from internal energy back to temperature
                    newT = newE * 2. * mu / 3. / k_B / n_B
                    
                    # Calculate neutral fractions
                    newHI = n_H - newHII
                    newHeI = n_He - newHeII - newHeIII
                             
                    # Update quantities in 'data' -> 'newdata'                
                    newdata["HIDensity"][cell] = newHI                                                                                            
                    newdata["HIIDensity"][cell] = n_H - newHI
                    newdata["HeIDensity"][cell] = newHeI
                    newdata["HeIIDensity"][cell] = newHeII
                    newdata["HeIIIDensity"][cell] = newHeIII
                    newdata["ElectronDensity"][cell] = (n_H - newHI) + newHeIII + 2.0 * newHeIII
                    newdata["Temperature"][cell] = newT                   
                                                                                
                    cell_pack += dc
                                                    
                    ######################################
                    ################ DONE ################     
                    ######################################                                   
                
            if self.HIIRestrictedTimestep or self.HeIIRestrictedTimestep or self.HeIIIRestrictedTimestep:
                for cell in self.grid:
                    dtphot[cell] = self.ComputePhotonTimestep(newdata, cell, self.rs.BolometricLuminosity(t), n_H_arr[0], n_He_arr[0])            
        
        # If multiple processors at work, communicate data and timestep                                                                                          
        if (size > 1) and (self.ParallelizationMethod == 1):
            for key in newdata.keys(): 
                newdata[key] = MPI.COMM_WORLD.allreduce(newdata[key], newdata[key])
                
            dtphot = MPI.COMM_WORLD.allreduce(dtphot, dtphot) 
                                
        if self.HIIRestrictedTimestep or self.HeIIRestrictedTimestep or self.HeIIIRestrictedTimestep: 
            newdt = min(np.min(dtphot), 2 * dt)
        else: 
            newdt = dt
        
        if self.LightCrossingTimeRestrictedStep: 
            newdt = min(newdt, self.LightCrossingTimeRestrictedStep * self.LengthUnits / self.GridDimensions / 29979245800.0)
        
        if rank == 0 and self.ProgressBar: 
            pbar.finish()
        
        # Update photon packages        
        if not self.InfiniteSpeedOfLight: 
            newdata['PhotonPackages'] = self.UpdatePhotonPackages(packs, t + dt, newdata)  # t + newdt?            
                                
        # Store timestep information
        newdata['dtPhoton'] = dtphot                        
                                
        if size > 1: 
            lb = self.control.LoadBalance(dtphot)   
        else: 
            lb = None     
                                                                                                                                                     
        return newdata, h, newdt, lb 
        
    def RateEquations(self, q, t, args):    
        """
        NEW!
        
        args = (nabs, n_H, n_He, n_e, Gamma, gamma, Beta, alpha, k_H, zeta, eta, psi)
        
        where Gamma, gamma, beta, alpha = 3 element arrays (one entry per species)
        
        """
                
        nabs = args[0]
        n_H = args[1]
        n_He = args[2]
        n_e = args[3]
        
        Gamma = args[4]
        gamma = args[5]
        Beta = args[6]
        alpha = args[7]
        k_H = args[8]
        zeta = args[9]
        eta = args[10]
        psi = args[11]
        
        q[0] = (Gamma[0] + gamma[0]) * nabs[0] + Beta[0] * n_e * nabs[0] - alpha[0] * n_e * q[0]
                
        #q[1] = Gamma_HeI * n_HeI + Beta_HeI * n_e * n_HeI - Beta_HeII * n_e * q[1] - \
        #              alpha_HeII * n_e * q[1] + alpha_HeIII * n_e * n_HeIII - xi_HeII * n_e * q[1]    
        #q[2] = Gamma_HeII * n_HeII + Beta_HeII * n_e * n_HeII - alpha_HeIII * n_e * q[2]      
        #
        #
        
        
        
        q[3] = np.sum(k_H * nabs) - n_e * (np.sum(zeta * nabs) + np.sum(eta * nabs) + np.sum(psi * nabs))
        
        return q    
    
    def qdot(self, q, t, *args):
        """
        This function returns the right-hand side of our ODE's (Equations 1, 2, 3 and 9 in Mirocha et al. 2012).

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
        indices = args[0][7]
                        
        # Derived quantities
        n_HII = min(q[0], n_H)  # This could be > n_H within machine precision and really screw things up
        n_HI = n_H - n_HII
        x_HII = n_HII / n_H   
              
        n_HeII = q[1]
        n_HeIII = q[2]
        n_HeI = n_He - (n_HeII + n_HeIII) 
        
        n_e = n_HII + n_HeII + 2.0 * n_HeIII
        nion = [n_HII, n_HeII, n_HeIII]
        nabs = [n_H, n_HeI, n_HeII]
        n_B = n_H + n_He + n_e

        E = q[3]        
        if self.Isothermal: 
            T = self.InitialTemperature
        else: 
            T = E * 2. * mu / 3. / k_B / n_B
                
        # First, solve for rate coefficients
        alpha_HII = 2.6e-13 * (T / 1.e4)**-0.85    
        Gamma_HI = self.IonizationRateCoefficientHI(ncol, n_e, n_HI, n_HeI, x_HII, T, r, Lbol, indices)        
                                                                                             
        if self.MultiSpecies > 0: 
            Gamma_HeI = self.IonizationRateCoefficientHeI(ncol, n_HI, n_HeI, x_HII, T, r, Lbol, indices)
            Gamma_HeII = self.IonizationRateCoefficientHeII(ncol, n_HI, n_HeI, x_HII, T, r, Lbol, indices)
            Beta_HeI = 2.38e-11 * np.sqrt(T) * (1. + np.sqrt(T / 1.e5))**-1. * np.exp(-2.853e5 / T) * self.CollisionalIonization
            Beta_HeII = 5.68e-12 * np.sqrt(T) * (1. + np.sqrt(T / 1.e5))**-1. * np.exp(-6.315e5 / T) * self.CollisionalIonization
            alpha_HeII = 9.94e-11 * T**-0.6687                                                           
            alpha_HeIII = 3.36e-10 * T**-0.5 * (T / 1e3)**-0.2 * (1. + (T / 4.e6)**0.7)**-1        # To n >= 1
            if T < 2.2e4: alpha_HeIII *= (1.11 - 0.044 * np.log(T))                                # To n >= 2
            else: alpha_HeIII *= (1.43 - 0.076 * np.log(T))
            xi_HeII = 1.9e-3 * T**-1.5 * np.exp(-4.7e5 / T) * (1. + 0.3 * np.exp(-9.4e4 / T))
        else: 
            Gamma_HeI = Gamma_HeII = Beta_HeI = Beta_HeII = alpha_HeII = alpha_HeIII = alpha_HeIII = xi_HeII = 0
                                                                
        # Always solve hydrogen rate equation
        q[0] = Gamma_HI * n_HI - alpha_HII * n_e * q[0]
       
        # Only solve helium rate equations if self.MultiSpeces = 1
        if self.MultiSpecies:
            q[1] = Gamma_HeI * n_HeI + Beta_HeI * n_e * n_HeI - Beta_HeII * n_e * q[1] - \
                      alpha_HeII * n_e * q[1] + alpha_HeIII * n_e * n_HeIII - xi_HeII * n_e * q[1]    
            q[2] = Gamma_HeII * n_HeII + Beta_HeII * n_e * n_HeII - alpha_HeIII * n_e * q[2]            

        # Only solve internal energy equation if we're not doing an isothermal calculation
        if not self.Isothermal:
            q[3] = self.HeatGain(ncol, nabs, x_HII, r, Lbol, indices) - \
                self.HeatLoss(nabs, nion, n_e, n_B, E * 2. * mu / 3. / k_B / n_B, z, mu)                                
                                                                        
        return q     

    def UpdatePhotonPackages(self, packs, t_next, data):
        """
        Get rid of old photon packages.
        """    
                
        to_eliminate = []
        for i, pack in enumerate(packs):
            if (t_next - pack[0]) > (self.LengthUnits / c): to_eliminate.append(i)
            
        to_eliminate.reverse()    
        for element in to_eliminate: packs.pop(element)
                
        return packs
        
