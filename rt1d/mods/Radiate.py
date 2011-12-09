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
from rt1d.mods.SolveRateEquations import SolveRateEquations
from Integrate import simpson as integrate

try:
    from progressbar import *
    pb = True
except ImportError:
    "Module progressbar not found."
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
        self.CellCrossingTime = self.dx / c
        self.HIColumn = n_col[0]
        self.HeIColumn = n_col[1]
        self.HeIIColumn = n_col[2]
        
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
        indices = args[0][7]
                        
        # Derived quantities
        n_HII = min(q[0], n_H)          # This could be > n_H within machine precision and really screw things up
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
        if self.Isothermal: T = self.InitialTemperature
        else: T = E * 2. * mu / 3. / k_B / n_B
                
        # First, solve for rate coefficients
        alpha_HII = 2.6e-13 * (T / 1.e4)**-0.85    
        Gamma_HI = self.IonizationRateCoefficientHI(ncol, n_e, n_HI, n_HeI, x_HII, T, r, Lbol, indices)        
                                                                                             
        if self.MultiSpecies > 0: 
            Gamma_HeI = self.IonizationRateCoefficientHeI(ncol, n_HI, n_HeI, x_HII, T, r, Lbol, indices)
            Gamma_HeII = self.IonizationRateCoefficientHeII(ncol, x_HII, r, Lbol, indices)
            Beta_HeI = 2.38e-11 * np.sqrt(T) * (1. + np.sqrt(T / 1.e5))**-1. * np.exp(-2.853e5 / T) * self.CollisionalIonization
            Beta_HeII = 5.68e-12 * np.sqrt(T) * (1. + np.sqrt(T / 1.e5))**-1. * np.exp(-6.315e5 / T) * self.CollisionalIonization
            alpha_HeII = 9.94e-11 * T**-0.48                                                            
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

    def EvolvePhotons(self, data, t, dt, h, lb):
        """
        This routine calls our solvers and updates 'data'.
        """
        
        # Do this so an MPI all-reduce doesn't add stuff together
        if self.ParallelizationMethod == 1 and size > 1:
            solve_arr = np.arange(len(self.grid))
            proc_mask = np.zeros_like(solve_arr)
            
            condition = (solve_arr >= lb[rank]) & (solve_arr < lb[rank + 1])
            proc_mask[condition] = 1
            solve_arr = solve_arr[proc_mask == 1]   
                            
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
        else: n_He_arr = x_HeI_arr = x_HeII_arr = x_HeIII_arr = np.zeros_like(x_HI_arr)
                                                        
        # If we're in an expanding universe, dilute densities by (1 + z)**3    
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
        
        # Initialize dtphot array
        if self.HIIRestrictedTimestep: 
            dtphot = np.zeros(len(self.grid))
            dtphot[0:self.StartCell] = 1e50
                
        ###
        ## SOLVE: c -> inf
        ###        
        if self.InfiniteSpeedOfLight:
            
            # Loop over cells radially, solve rate equations, update values in data -> newdata
            for cell in self.grid:
                            
                # If within our buffer zone (where we don't solve rate equations), continue
                if cell < self.StartCell: continue
                
                # If this cell belongs to another processor, continue
                if self.ParallelizationMethod == 1 and size > 1:
                    if cell not in solve_arr: continue
                        
                if rank == 0 and self.ProgressBar: pbar.update(cell * size)
                                                                
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
                                        
                # For convenience - dunno if we need to compute this stuff anymore
                ncol = ncol_all[cell]
                nabs = nabs_all[cell]
                nion = nion_all[cell]
                n_H = n_HI + n_HII
                n_He = n_HeI + n_HeII + n_HeIII                
                n_B = n_H + n_He + n_e
                                                        
                # Compute internal energy for this cell
                T = data["Temperature"][cell]
                E = 3. * k_B * T * n_B / mu / 2.
                        
                # Read radius
                r = self.r[cell]
                                            
                ######################################
                ######## Solve Rate Equations ########
                ######################################
                
                # Retrieve indices used for 3D interpolation
                indices = None
                if self.MultiSpecies > 0: 
                    indices = self.Interpolate.GetIndices3D(ncol)
                           
                # Call solver                                    
                tarr, qnew, h = self.solver.integrate(self.qdot, q_all[cell], t, t + dt, None, h, \
                    r, z, mu, n_H, n_He, ncol, Lbol, indices)
                                                                                         
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
                                       
                # Could probably do this with data, and it wouldn't matter too much - would be 
                # a more conservative estimate of Gamma_i actually.                                
                if self.HIIRestrictedTimestep: 
                    dtphot[cell] = self.ComputePhotonTimestep(newdata, cell, Lbol, n_H, n_He)                            
                                
                ######################################
                ################ DONE ################
                ######################################
                
        ###
        ## SOLVE: c = finite
        ###   
        else:                
                                          
            if self.HIIRestrictedTimestep: 
                dtphot = 1e50 * np.ones_like(self.grid)    
                                    
            # Loop over photon packages, updating values in cells: data -> newdata
            for j, pack in enumerate(packs):
                t_birth = pack[0]
                r_pack = (t - t_birth) * c        # Position of package before evolving photons
                r_max = r_pack + dt * c           # Furthest this package will get this timestep
                                                                                 
                # Cells we need to know about - not necessarily integer
                cell_pack = r_pack * self.GridDimensions / self.LengthUnits
                cell_pack_max = r_max * self.GridDimensions / self.LengthUnits
                
                if cell_pack_max < self.StartCell: continue
                              
                Lbol = pack[-1] / pack[1]          
                                                
                # Advance this photon package as far as it will go on this global timestep  
                while cell_pack < cell_pack_max:
                                                            
                    # What cell are we in
                    cell = int(round(cell_pack))
                    
                    if cell >= self.GridDimensions: break
                    
                    # Compute dc (like dx but in fractional cell units)
                    if cell_pack % 1 == 0: dc = min(cell_pack_max - cell_pack, 1)
                    else: dc = min(math.ceil(cell_pack) - cell_pack, cell_pack_max - cell_pack)        
                                              
                    if cell < self.StartCell: 
                        cell_pack += dc
                        continue
                                                
                    # We really need to evolve this cell until the next photon package arrives, which
                    # is probably longer than a cell crossing time unless the global dt is vv small.
                    if (len(packs) > 1) and ((j + 1) < len(packs)): subdt = min(dt, packs[j + 1][0] - pack[0])
                    else: subdt = dt
                                                                                                                                                                                                                                                                                                                                              
                    r = cell_pack * self.LengthUnits / self.GridDimensions
                    
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
                    
                    mu = 1. / (self.X * (1. + x_HII) + self.Y * (1. + x_HeII + x_HeIII) / 4.)
                                            
                    # For convenience     
                    nabs = [n_HI, n_HeI, n_HeII]
                    nion = [n_HII, n_HeII, n_HeIII]
                    n_H = n_HI + n_HII
                    n_He = n_HeI + n_HeII + n_HeIII
                    n_B = n_H + n_He + n_e              # STILL DONT UNDERSTAND THIS
                                            
                    # Compute internal energy for this cell
                    T = newdata["Temperature"][cell]
                    E = 3. * k_B * T * n_B / mu / 2.
                    
                    packs[j][2] += newdata['HIDensity'][cell] * dc * self.CellCrossingTime * c  
                    packs[j][3] += newdata['HeIDensity'][cell] * dc * self.CellCrossingTime * c 
                    packs[j][4] += newdata['HeIIDensity'][cell] * dc * self.CellCrossingTime * c
                    
                    ######################################
                    ######## Solve Rate Equations ########
                    ######################################
                    
                    # Retrieve indices used for 3D interpolation
                    indices = None
                    if self.MultiSpecies > 0: 
                        indices = self.Interpolate.GetIndices3D(ncol)
                    
                    values = (n_HII, n_HeII, n_HeIII, E)
                    qnew = [[0, n_HII], [0, n_HeII], [0, n_HeIII], [0, E]]
                                                                                
                    tarr, qnew, h = self.solver.integrate(self.qdot, values, t, t + subdt, None, h, \
                        r, z, mu, n_H, n_He, packs[j][2:5], Lbol, indices)
                                        
                    # Unpack results of coupled equations - Remember: these are lists and we only need the last entry 
                    newHII, newHeII, newHeIII, newE = qnew

                    # Convert from internal energy back to temperature
                    newT = newE[-1] * 2. * mu / 3. / k_B / n_B
                             
                    # Update quantities in 'data' -> 'newdata'     
                    newdata["HIDensity"][cell] = newHI                                                                   
                    newdata["HIIDensity"][cell] = n_H - newHI
                    newdata["HeIDensity"][cell] = newHeI
                    newdata["HeIIDensity"][cell] = newerHeII
                    newdata["HeIIIDensity"][cell] = newerHeIII
                    newdata["ElectronDensity"][cell] = (n_H - newHI) + newerHeII + 2.0 * newerHeIII
                    newdata["Temperature"][cell] = newT                               
                                                                                
                    cell_pack += dc
                                                    
                    ######################################
                    ################ DONE ################     
                    ######################################                                   
                
            if self.HIIRestrictedTimestep or self.HeIIRestrictedTimestep or self.HeIIIRestrictedTimestep:
                for cell in self.grid[self.StartCell:]:
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
        
        if rank == 0 and self.ProgressBar: pbar.finish()
        
        # Update photon packages        
        if not self.InfiniteSpeedOfLight: 
            newdata['PhotonPackages'] = self.UpdatePhotonPackages(packs, t + dt, newdata)  # t + newdt?            
                                
        if size > 1: lb = self.LoadBalance(dtphot)   
        else: lb = None     
                                                                                                           
        return newdata, h, newdt, lb  
        
    def ComputePhotonTimestep(self, newdata, cell, Lbol, n_H, n_He):
        """
        Use Shapiro et al. 2004 criteria to set next timestep.
        
        Pre-compute Gamma, alpha, etc. to save time?
        
        """          
    
        ncol = [np.cumsum(newdata["HIDensity"])[cell] * self.dx, np.cumsum(newdata["HeIDensity"])[cell] * self.dx,
                np.cumsum(newdata["HeIIDensity"])[cell] * self.dx]
        
        T = newdata['Temperature'][cell]
        n_e = newdata["ElectronDensity"][cell]
        n_HI = newdata["HIDensity"][cell]
        n_HII = newdata["HIDensity"][cell]
        x_HII = newdata['HIIDensity'][cell] / n_H 
        n_HeI = newdata["HeIDensity"][cell]
        n_HeII = newdata["HeIIDensity"][cell]
                                
        if self.MultiSpecies: indices = self.Interpolate.GetIndices3D(ncol) 
        else: indices = None        
         
        dtHI = 1e50        
        tauHI = self.Interpolate.interp(indices, "TotalOpticalDepth0", ncol)
        if tauHI >= 0.5:

            Gamma = self.IonizationRateCoefficientHI(ncol, n_e, n_HI, n_HeI, 
                x_HII, T, self.r[cell], Lbol, indices)        
            alpha = 2.6e-13 * (T / 1.e4)**-0.85  
            
            # Shapiro et al. 2004
            dtHI = self.MaxHIIChange * newdata["HIDensity"][cell] / \
                np.abs(n_HI * Gamma - n_HII * n_e * alpha)
        
        dtHeI = 1e50
        if self.MultiSpecies and self.HeIIRestrictedTimestep:
            
            tauHeI = self.Interpolate.interp(indices, "TotalOpticalDepth1", ncol)
            print tauHeI
            if tauHeI >= 0.5:
                xHeII = n_HeII / n_He
                
                Beta = 2.38e-11 * np.sqrt(T) * (1. + np.sqrt(T / 1.e5))**-1. * np.exp(-2.853e5 /T) * self.CollisionalIonization
                Gamma = self.IonizationRateCoefficientHeI(ncol, n_HI, n_HeI, x_HII, T, self.r[cell], Lbol, indices)                    
                alpha = 9.94e-11 * T**-0.48   
                
                # Analogous to Shapiro et al. 2004 but for helium
                dtHeI = self.MaxHeIIChange * n_HeI / \
                    np.abs(n_HeI * (Gamma + n_e * Beta) - n_HeII * n_e * alpha) 
                
        dtHeII = 1e50
        if self.MultiSpecies and self.HeIIIRestrictedTimestep:
            pass
            #tauHeII = self.Interpolate.interp(indices, "TotalOpticalDepth2", ncol)
            #
            #if tauHeII >= 0.5:
            #    xHeIII = n_HeIII / n_He
            #    
            #    Beta = 5.68e-12 * np.sqrt(T) * (1. + np.sqrt(T / 1.e5))**-1. * np.exp(-6.315e5 / T)
            #    
            #    Beta = 2.38e-11 * np.sqrt(T) * (1. + np.sqrt(T / 1.e5))**-1. * np.exp(-2.853e5 /T) * self.CollisionalIonization
            #    Gamma = self.IonizationRateCoefficientHeII(ncol, n_HI, n_HeI, x_HII, T, self.r[cell], Lbol, indices)                    
            #    alpha = 9.94e-11 * T**-0.48   
            #    
            #    # Analogous to Shapiro et al. 2004 but for helium
            #    dtHeI = self.MaxHeIIChange * n_HeI / \
            #        np.abs(newdata["HeIDensity"][cell] * (Gamma + n_e * Beta) - \
            #        newdata['HeIIDensity'][cell] * n_e * alpha) 
        
        
        return min(dtHI, dtHeI, dtHeII)                   
        
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
        
    def LoadBalance(self, dtphot):
        """
        Return cells that should be solved by each processor.
        """    
        
        nsubsteps = 1. / dtphot[self.StartCell:]
                
        # Compute CDF for timesteps
        cdf = np.cumsum(nsubsteps) / np.sum(nsubsteps)
        intervals = np.linspace(1. / size, 1, size)
                
        lb = list(np.interp(intervals, cdf, self.grid[self.StartCell:], left = self.StartCell))
        lb.insert(0, self.StartCell)
                
        lb[-1] = self.GridDimensions
        
        for i, entry in enumerate(lb):
            lb[i] = int(entry)
        
        # Make sure no two elements are the same - this may not always work
        while np.any(np.diff(lb) == 0):
            for i, entry in enumerate(lb[1:-1]):
                                
                if entry == self.StartCell: 
                    lb[i + 1] = entry + 1
                if entry == lb[i]:
                    lb[i + 1] = entry + 1
                                                                        
        return lb
        
    def IonizationRateCoefficientHI(self, ncol, n_e, n_HI, n_HeI, x_HII, T, r, Lbol, indices):
        """
        Returns ionization rate coefficient for HI, which we denote elsewhere as Gamma_HI.  Includes photo, collisional, 
        and secondary ionizations from fast electrons.
        
            units: 1 / s
        """     
           
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
        
    def IonizationRateCoefficientHeI(self, ncol, n_HI, n_HeI, x_HII, T, r, Lbol, indices):
        """
        Returns ionization rate coefficient for HeI, which we denote elsewhere as Gamma_HeI.  Includes photo 
        and secondary ionizations from fast electrons.  Unlike the hydrogen case, the collisional ionizations
        are included in the rate equation itself instead of a coefficient.
        
            units: 1 / s
        """                
        
        IonizationRate = Lbol * \
                         self.Interpolate.interp(indices, "PhotoIonizationRate{0}".format(1), ncol)
        
        if self.SecondaryIonization:
            IonizationRate += Lbol * \
                              self.esec.DepositionFraction(0.0, x_HII, channel = 2) * \
                              self.Interpolate.interp(indices, "SecondaryIonizationRateHeI1", ncol)
            
            IonizationRate += (n_HI / n_HeI) * Lbol * \
                              self.esec.DepositionFraction(0.0, x_HII, channel = 2) * \
                              self.Interpolate.interp(indices, "SecondaryIonizationRateHeI1", ncol) 
        
        if not self.PlaneParallelField: IonizationRate /= (4. * np.pi * r**2)
        
        return IonizationRate
        
    def IonizationRateCoefficientHeII(self, ncol, x_HII, r, Lbol, indices):
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
                        
        if not self.PlaneParallelField: IonizationRate /= (4. * np.pi * r**2)
        
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
        Returns the total optical depth at energy E due to column densities of HI, HeI, and HeII, which
        are stored in the variable 'n' as a three element array.
        """
        
        func = lambda E: PhotoIonizationCrossSection(E, 0) * n[0] + PhotoIonizationCrossSection(E, 1) * n[1] \
            + PhotoIonizationCrossSection(E, 2) * n[2]
            
        return quad(func, self.rs.Emin, self.rs.Emax, epsrel = 1e-8)[0]   
            