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
    widget = ["rt1d: ", Percentage(), ' ', Bar(marker = RotatingMarker()), ' ', ETA(), ' ']
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

neglible_tau = 1e-12
neglible_column = 1

class Radiate:
    def __init__(self, pf, data, itabs, n_col): 
        self.pf = pf
        self.rs = RadiationSource(pf)
        self.cosmo = Cosmology(pf)
        self.coeff = RateCoefficients(pf, itabs = itabs, rs = self.rs, n_col = n_col)
        self.control = ControlSimulation(pf)
        self.debug = pf["Debug"]
        
        self.ProgressBar = pf["ProgressBar"] and pb
        
        # Methodology
        self.PhotonConserving = pf['PhotonConserving']
        self.ParallelizationMethod = pf["ParallelizationMethod"]        
        self.InterpolationMethod = pf["InterpolationMethod"]
        
        self.OutputRates = pf["OutputRates"]
        
        # Grid/units/etc
        self.GridDimensions = int(pf["GridDimensions"])
        self.grid = np.arange(self.GridDimensions)
        self.LengthUnits = pf["LengthUnits"]
        self.TimeUnits = pf["TimeUnits"]
        self.StopTime = pf["StopTime"] * self.TimeUnits
        self.StartRadius = pf["StartRadius"]
        self.StartCell = self.StartRadius * self.GridDimensions
        self.R0 = self.StartRadius * self.LengthUnits
        
        # Timestepping
        self.AdaptiveTimestep = pf["ODEAdaptiveStep"]
        self.HIRestrictedTimestep = pf["HIRestrictedTimestep"]
        self.HeIRestrictedTimestep = pf["HeIRestrictedTimestep"]
        self.HeIIRestrictedTimestep = pf["HeIIRestrictedTimestep"]
        self.HeIIIRestrictedTimestep = pf["HeIIIRestrictedTimestep"]
        self.ElectronFractionRestrictedTimestep = pf["ElectronFractionRestrictedTimestep"]
        self.AdaptiveGlobalStep = self.HIRestrictedTimestep or self.HeIRestrictedTimestep \
            or self.HeIIRestrictedTimestep or self.HeIIIRestrictedTimestep or self.ElectronFractionRestrictedTimestep
        self.LightCrossingTimeRestrictedTimestep = pf["LightCrossingTimeRestrictedTimestep"]
        self.AdaptiveODEStep = pf["ODEAdaptiveStep"]
        self.MaxStep = pf["ODEMaxStep"] * self.TimeUnits
        self.MinStep = pf["ODEMinStep"] * self.TimeUnits
        
        # Tolerance
        self.MinimumSpeciesFraction = pf["MinimumSpeciesFraction"]
        
        # Physics
        self.MultiSpecies = pf["MultiSpecies"]
        self.InfiniteSpeedOfLight = pf["InfiniteSpeedOfLight"]
        self.Isothermal = pf["Isothermal"]
        self.ComptonCooling = pf["ComptonCooling"]
        self.CollisionalIonization = pf["CollisionalIonization"]
        self.CollisionalExcitation = pf["CollisionalExcitation"]
        self.SecondaryIonization = pf["SecondaryIonization"]
        self.FreeFreeEmission = pf["FreeFreeEmission"]
        self.CosmologicalExpansion = pf["CosmologicalExpansion"]
        self.PlaneParallelField = pf["PlaneParallelField"]
        self.C = pf["ClumpingFactor"]
        
        self.InitialRedshift = pf["InitialRedshift"]
        self.InitialHydrogenDensity = (data["HIDensity"][0] + data["HIIDensity"][0]) / (1. + self.InitialRedshift)**3
        self.InitialHeliumDensity = (data["HeIDensity"][0] + data["HeIIDensity"][0] + data["HeIIIDensity"][0]) / (1. + self.InitialRedshift)**3
        
        # Deal with log-grid, compute dx
        if pf['LogarithmicGrid']:
            self.r = np.logspace(np.log10(self.StartRadius * self.LengthUnits), \
                np.log10(self.LengthUnits), self.GridDimensions + 1)
        else:
            rmin = self.StartRadius * self.LengthUnits
            self.r = np.linspace(rmin, self.LengthUnits, self.GridDimensions + 1)
        
        self.dx = np.diff(self.r)   
        self.r = self.r[0:-1] 
                    
        self.CellCrossingTime = self.dx / c
        self.OnePhotonPackagePerCell = pf["OnePhotonPackagePerCell"]
        
        # Initial guesses for ODE solver
        self.TotalHydrogen = data["HIDensity"][-1] + data["HIIDensity"][-1]
        self.TotalHelium = data["HeIDensity"][-1] + data["HeIIDensity"][-1] + data["HeIIIDensity"][-1]
        guesses = [self.TotalHydrogen, 
                   self.TotalHelium, 
                   self.TotalHelium, 
                   3 * self.pf["InitialTemperature"] * k_B * (self.TotalHydrogen + self.TotalHelium) / 2.]
                
        # Initialize solver        
        self.solver = SolveRateEquations(pf, guesses = guesses, AdaptiveODEStep = self.AdaptiveODEStep, 
            hmin = self.MinStep, hmax = self.MaxStep, Dfun = None, maxiter = pf["ODEMaxIter"])                 
        
        # Initialize helium abundance                        
        self.Y = 0.2477 * self.MultiSpecies
        self.X = 1. - self.Y

        # For convenience 
        self.zeros_tmp = np.zeros(4)
        self.zeros_eq = np.zeros(4)
        
    def EvolvePhotonsAtFiniteSpeed(self):
        pass
    def EvolvePhotonsAtInfiniteSpeed(self):
        pass        

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
             
        # Convert time to redshift     
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

        # Compute column densities - meaning column density *between source and cell*
        ncol_HI = np.roll(np.cumsum(data["HIDensity"] * self.dx), 1)
        ncol_HeI = np.roll(np.cumsum(data["HeIDensity"] * self.dx), 1)
        ncol_HeII = np.roll(np.cumsum(data["HeIIDensity"] * self.dx), 1)
        ncol_HI[0] = ncol_HeI[0] = ncol_HeII[0] = neglible_column
        
        # Convenience arrays for column, absorber, and ion densities, plus some others
        ncol_all = np.transpose(np.log10([ncol_HI, ncol_HeI, ncol_HeII]))
        nabs_all = np.transpose([data["HIDensity"], data["HeIDensity"], data["HeIIDensity"]])
        nion_all = np.transpose([data["HIIDensity"], data["HeIIDensity"], data["HeIIIDensity"]])
        mu_all = 1. / (self.X * (1. + x_HII_arr) + self.Y * (1. + x_HeII_arr + x_HeIII_arr) / 4.)
        nB_all = n_H_arr + n_He_arr + data["ElectronDensity"]
        q_all = np.transpose([data["HIIDensity"], data["HeIIDensity"], data["HeIIIDensity"], 3. * k_B * data["Temperature"] * nB_all / mu_all / 2.])
        
        # Retrieve indices used for 1-3D interpolation
        indices_all = []
        for i, col in enumerate(ncol_all):
            if self.MultiSpecies > 0: 
                indices_all.append(self.coeff.Interpolate.GetIndices3D(col))
            else:
                indices_all.append(None)
        
        # Compute optical depths *to* all cells
        # Only use this for timestep calculation?
        tau_all_arr = np.zeros([3, self.GridDimensions])    
        if not self.pf['TabulateIntegrals']:
            sigma0 = PhotoIonizationCrossSection(self.rs.E, species = 0)
            tmp_nHI = np.transpose(len(self.rs.E) * [ncol_HI])            
            tau_all_arr[0] = np.sum(tmp_nHI * sigma0, axis = 1)
                        
            if self.MultiSpecies:
                sigma1 = PhotoIonizationCrossSection(self.rs.E, species = 1)
                sigma2 = PhotoIonizationCrossSection(self.rs.E, species = 2)
                tmp_nHeI = np.transpose(len(self.rs.E) * [ncol_HeI])
                tmp_nHeII = np.transpose(len(self.rs.E) * [ncol_HeII])
                tau_all_arr[1] = np.sum(tmp_nHeI * sigma1, axis = 1)
                tau_all_arr[2] = np.sum(tmp_nHeII * sigma2, axis = 1)
            
        else:
            for i, col in enumerate(np.log10(ncol_HI)):
                tau_all_arr[0][i] = self.coeff.Interpolate.OpticalDepth(col, 0)
                
                if self.MultiSpecies:
                    tau_all_arr[1][i] = self.coeff.Interpolate.OpticalDepth(np.log10(ncol_HeI[i]), 1)
                    tau_all_arr[2][i] = self.coeff.Interpolate.OpticalDepth(np.log10(ncol_HeII[i]), 2)
                    
            tau_all_arr[0][0] = tau_all_arr[1][0] = tau_all_arr[2][0] = neglible_tau 

        tau_all = zip(*tau_all_arr)         
                                
        # Print status, and update progress bar
        if rank == 0: 
            print "rt1d: %g < t < %g" % (t / self.TimeUnits, (t + dt) / self.TimeUnits)
        if rank == 0 and self.ProgressBar: 
            pbar = ProgressBar(widgets = widget, maxval = self.grid[-1]).start()
        
        # If accreting black hole, luminosity will change with time.
        Lbol = self.rs.BolometricLuminosity(t)
        
        # Set up 'packs' structure for c < infinity runs
        if not self.InfiniteSpeedOfLight: 
            
            # PhotonPackage guide: pack = [EmissionTime, EmissionTimeInterval, ncolHI, ncolHeI, ncolHeII, Energy]
            
            # Photon packages going from oldest to youngest - will have to create it on first timestep
            try: 
                packs = list(data['PhotonPackages']) 
            except KeyError: 
                packs = []            
            
            if packs and self.OnePhotonPackagePerCell:
                t_packs = np.array(zip(*packs)[0])    # Birth times for all packages
                r_packs = (t - t_packs) * c
                
                # If there are already photon packages in the first cell, add energy in would-be packet to preexisting one
                if r_packs[-1] < (self.StartRadius * self.LengthUnits):
                    packs[-1][-1] += Lbol * dt
                else:
                    # Launch new photon package - [t_birth, ncol_HI_0, ncol_HeI_0, ncol_HeII_0]                
                    packs.append(np.array([t, dt, 0., 0., 0., Lbol * dt]))
            else: 
                packs.append(np.array([t, dt, 0., 0., 0., Lbol * dt]))
        
        # Initialize timestep array
        if self.AdaptiveGlobalStep: 
            if self.InfiniteSpeedOfLight:
                dtphot = 1. * np.zeros_like(self.grid)
            else:
                dtphot = 1.e50 * np.ones_like(self.grid)
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
                ncol = ncol_all[cell]   # actully log10(ncol)
                nabs = nabs_all[cell]
                nion = nion_all[cell]
                n_H = n_HI + n_HII
                n_He = n_HeI + n_HeII + n_HeIII                
                n_B = n_H + n_He + n_e
                
                # Read in mean molecular weight for this cell
                mu = mu_all[cell] 
                                                                              
                # Read in temperature and internal energy for this cell
                T = data["Temperature"][cell]
                E = q_all[cell] 
                        
                # Read radius
                r = self.r[cell]
                                
                # Retrieve indices used for 3D interpolation
                indices = indices_all[cell]
                
                # Retrieve optical depth to this cell
                tau = tau_all[cell]
                                
                # Retrieve path length through this cell
                dx = self.dx[cell]     
                                                                             
                # Retrieve coefficients and what not.
                args = [nabs, nion, n_H, n_He, n_e]
                args.extend(self.coeff.ConstructArgs(args, indices, Lbol, r, ncol, T, dx, t))
                
                # Unpack so we have everything by name
                nabs, nion, n_H, n_He, n_e, Gamma, gamma, Beta, alpha, k_H, zeta, eta, psi, xi, omega = args                                                          
                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
                ######################################
                ######## Solve Rate Equations ########
                ######################################        
                                                              
                tarr, qnew, h, odeitr, rootitr = self.solver.integrate(self.RateEquations, q_all[cell], t, t + dt, 
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
                newdata["HIIDensity"][cell] = newHII
                newdata["HeIDensity"][cell] = newHeI
                newdata["HeIIDensity"][cell] = newHeII
                newdata["HeIIIDensity"][cell] = newHeIII
                newdata["ElectronDensity"][cell] = newHII + newHeII + 2.0 * newHeIII
                newdata["Temperature"][cell] = newT    
                newdata["OpticalDepth"][cell] = tau
                newdata["ODEIterations"][cell] = odeitr
                newdata["ODEIterationRate"][cell] = odeitr / (h / self.TimeUnits)
                newdata["RootFinderIterations"][cell] = rootitr
                
                if self.OutputRates:
                    for i in xrange(3):
                        newdata['PhotoIonizationRate%i' % i][cell] = Gamma[i]
                        newdata['PhotoHeatingRate%i' % i][cell] = k_H[i]
                        newdata['CollisionalIonizationRate%i' % i][cell] = Beta[i] 
                        newdata['RadiativeRecombinationRate%i' % i][cell] = alpha[i] 
                        newdata['CollisionalIonzationCoolingRate%i' % i][cell] = zeta[i] 
                        newdata['RecombinationCoolingRate%i' % i][cell] = eta[i] 
                        newdata['CollisionalExcitationCoolingRate%i' % i][cell] = psi[i]
                        
                        newdata['SecondaryIonizationRate%i' % i][cell] = gamma[i] 
                        
                        if i == 2:
                            newdata['DielectricRecombinationRate'][cell] = xi[i]
                            newdata['DielectricRecombinationCoolingRate'][cell] = omega[i]
                                                                                                  
                ######################################
                ################ DONE ################
                ######################################
                    
                # Adjust timestep based on maximum allowed neutral fraction change     
                if self.AdaptiveGlobalStep:
                    dtphot[cell] = self.control.ComputePhotonTimestep(tau, 
                        nabs, nion, ncol, n_H, n_He, n_e, n_B, Gamma, gamma, Beta, alpha, xi, dt) 

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
                    
                    if cell >= self.GridDimensions: 
                        break
                    
                    # Compute dc (like dx but in fractional cell units)
                    if cell_pack % 1 == 0: 
                        dc = min(cell_pack_max - cell_pack, 1)
                    else: 
                        dc = min(math.ceil(cell_pack) - cell_pack, cell_pack_max - cell_pack)        
                                                         
                    # We really need to evolve this cell until the next photon package arrives, which
                    # is probably longer than a cell crossing time unless the global dt is vv small.
                    if (len(packs) > 1) and ((j + 1) < len(packs)): 
                        subdt = min(dt, packs[j + 1][0] - pack[0])
                    else: 
                        subdt = dt
                                                                                                                                                                                                                                                                                                                                              
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
                    mu = mu_all[cell] 
                    
                    # Retrieve path length through this cell
                    dx = self.dx[cell]     
                                            
                    # For convenience     
                    nabs = [n_HI, n_HeI, n_HeII]
                    nion = [n_HII, n_HeII, n_HeIII]
                    n_H = n_HI + n_HII
                    n_He = n_HeI + n_HeII + n_HeIII
                    n_B = n_H + n_He + n_e
                    
                    # Compute internal energy for this cell
                    T = newdata["Temperature"][cell]
                    E = 3. * k_B * T * n_B / mu / 2.
                    
                    q_cell = [n_HII, n_HeII, n_HeIII, E]
                    
                    # Crossing time
                    dct = self.CellCrossingTime
                    
                    # Add columns of this cell
                    packs[j][2] += newdata['HIDensity'][cell] * dc * dct * c  
                    packs[j][3] += newdata['HeIDensity'][cell] * dc * dct * c 
                    packs[j][4] += newdata['HeIIDensity'][cell] * dc * dct * c
                    ncol = packs[j][2:5]
                    
                    ######################################
                    ######## Solve Rate Equations ########
                    ######################################
                    
                    # Retrieve indices used for 3D interpolation
                    indices = None
                    if self.MultiSpecies > 0: 
                        indices = self.coeff.Interpolate.GetIndices3D(ncol)
                    
                    # Retrieve coefficients and what not.
                    args = [nabs, nion, n_H, n_He, n_e]
                    args.extend(self.coeff.ConstructArgs(args, indices, Lbol, r, ncol, T, dx, dt, t))                    
                    
                    # Unpack so we have everything by name
                    nabs, nion, n_H, n_He, n_e, Gamma, gamma, Beta, alpha, k_H, zeta, eta, psi, xi = args
                                                                                                             
                    ######################################
                    ######## Solve Rate Equations ########
                    ######################################                          
                                    
                    tarr, qnew, h, odeitr, rootitr = self.solver.integrate(self.RateEquations, q_cell, t, t + dt, 
                        None, h, *args)
                                        
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
                
            # Adjust timestep based on maximum allowed neutral fraction change     
            if self.AdaptiveGlobalStep:
                for cell in self.grid:
                    dtphot[cell] = self.control.ComputePhotonTimestep(tau, 
                        nabs, nion, ncol, n_H, n_He, n_e, n_B, Gamma, gamma, Beta, alpha, xi, dt) 
    
        # If multiple processors at work, communicate data and timestep                                                                                          
        if (size > 1) and (self.ParallelizationMethod == 1):
            for key in newdata.keys(): 
                newdata[key] = MPI.COMM_WORLD.allreduce(newdata[key], newdata[key])
                
            dtphot = MPI.COMM_WORLD.allreduce(dtphot, dtphot) 
                                
        if self.AdaptiveGlobalStep: 
            newdt = min(np.min(dtphot), 2 * dt)
        else: 
            newdt = dt
        
        if self.LightCrossingTimeRestrictedTimestep: 
            newdt = min(newdt, self.LightCrossingTimeRestrictedTimestep * self.LengthUnits / self.GridDimensions / c)
        
        if rank == 0 and self.ProgressBar: 
            pbar.finish()
        
        # Update photon packages        
        if not self.InfiniteSpeedOfLight: 
            newdata['PhotonPackages'] = self.UpdatePhotonPackages(packs, t + dt, newdata)  # t + newdt?            
                                
        # Store timestep information
        newdata['dtPhoton'] = dtphot
                                
        # Load balance grid                        
        if size > 1 and self.ParallelizationMethod == 1: 
            lb = self.control.LoadBalance(dtphot)   
        else: 
            lb = None      
                                                                                                                                                                                 
        return newdata, h, newdt, lb 
        
    def RateEquations(self, q, t, args):    
        """
        This function returns the right-hand side of our ODE's (Equations 1, 2, 3 and 9 in Mirocha et al. 2012).

        q = [n_HII, n_HeII, n_HeIII, E] - our four coupled equations. q = generalized quantity I guess.
        
        args = (nabs, nion, n_H, n_He, n_e, Gamma, gamma, Beta, alpha, k_H, zeta, eta, psi, xi, omega)
        
        where nabs, nion, Gamma, gamma, Beta, alpha, k_H, zeta, eta, psi, xi = 3 element arrays 
            (one entry per species)
            
        returns ionizations / second / cm^3
        
        """
                
        nabs = args[0]
        nion = args[1]
        n_H = args[2]
        n_He = args[3]
        n_e = args[4]
        
        Gamma = args[5]
        gamma = args[6]
        Beta = args[7]
        alpha = args[8]
        k_H = args[9]
        zeta = args[10]
        eta = args[11]
        psi = args[12]
        xi = args[13]
        omega = args[14]
                
        dqdt = self.zeros_tmp
        
        # Neutrals (current)
        nHI = n_H - q[0]
        nHeI = n_He - q[1] - q[2]
        nHeII = q[1]
                
        # Always solve hydrogen rate equation
        dqdt[0] = (Gamma[0] + Beta[0] * n_e) * nHI + \
                  (gamma[0][0] * nHI + gamma[0][1] * nHeI + gamma[0][2] * nHeII) - \
                   alpha[0] * self.C * n_e * q[0]        
                
        # Helium rate equations  
        if self.MultiSpecies:       
            dqdt[1] = (Gamma[1] + Beta[1] * n_e) * nHeI + \
                      (gamma[1][0] * nHI + gamma[1][1] * nHeI + gamma[1][2] * nHeII)  + \
                       alpha[2] * n_e * q[2] - \
                      (Beta[1] + alpha[1] + xi[1]) * n_e * q[1]
                              
            dqdt[2] = (Gamma[2] + Beta[2] * n_e) * q[1] - alpha[2] * n_e * q[2]
        
        # Temperature evolution - using np.sum is slow :(
        if not self.Isothermal:
            dqdt[3] = np.sum(k_H * nabs) - n_e * (np.sum(zeta * nabs) \
                + np.sum(eta * nabs) + np.sum(psi * nabs) + q[2] * omega[1])

        return dqdt

    def UpdatePhotonPackages(self, packs, t_next, data):
        """
        Get rid of old photon packages.
        """    
                
        to_eliminate = []
        for i, pack in enumerate(packs):
            if (t_next - pack[0]) > (self.LengthUnits / c): 
                to_eliminate.append(i)
            
        to_eliminate.reverse()    
        for element in to_eliminate: 
            packs.pop(element)
                
        return packs
        
