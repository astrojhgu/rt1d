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

from .Constants import *
from .Cosmology import Cosmology
from .ComputeCrossSections import PhotoIonizationCrossSection
from .RadiationSource import RadiationSource
from .SecondaryElectrons import SecondaryElectrons
from .Interpolate import Interpolate
from .ComputeRateCoefficients import RateCoefficients
from .SolveRateEquations import SolveRateEquations
from .ControlSimulation import ControlSimulation

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
        self.cosm = Cosmology(pf)
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
        
        self.MultiSpecies = pf["MultiSpecies"]
        
        # Timestepping
        self.AdaptiveTimestep = pf["ODEAdaptiveStep"]
        self.HIRestrictedTimestep = pf["HIRestrictedTimestep"]
        self.HeIRestrictedTimestep = pf["HeIRestrictedTimestep"]
        self.HeIIRestrictedTimestep = pf["HeIIRestrictedTimestep"]
        self.HeIIIRestrictedTimestep = pf["HeIIIRestrictedTimestep"]
        self.ElectronRestrictedTimestep = pf["ElectronRestrictedTimestep"]
        self.LightCrossingTimeRestrictedTimestep = pf["LightCrossingTimeRestrictedTimestep"]
        self.RedshiftRestrictedTimestep = pf["RedshiftRestrictedTimestep"]
        self.MaxRedshiftStep = pf["MaxRedshiftStep"]
        self.AdaptiveODEStep = pf["ODEAdaptiveStep"]
        self.MaxStep = pf["ODEMaxStep"] * self.TimeUnits
        self.MinStep = pf["ODEMinStep"] * self.TimeUnits
        
        self.AdaptiveGlobalStep = self.HIRestrictedTimestep or self.ElectronRestrictedTimestep
        if self.MultiSpecies:
            self.AdaptiveGlobalStep |= (self.HeIRestrictedTimestep or \
                self.HeIIRestrictedTimestep or self.HeIIIRestrictedTimestep)
        
        # Tolerance
        self.MinimumSpeciesFraction = pf["MinimumSpeciesFraction"]
        
        # Physics
        self.MultiSpecies = pf["MultiSpecies"]
        self.InfiniteSpeedOfLight = pf["InfiniteSpeedOfLight"]
        self.Isothermal = pf["Isothermal"]
        self.ComptonHeating = pf["ComptonHeating"]
        self.CollisionalIonization = pf["CollisionalIonization"]
        self.CollisionalExcitation = pf["CollisionalExcitation"]
        self.SecondaryIonization = pf["SecondaryIonization"]
        self.FreeFreeEmission = pf["FreeFreeEmission"]
        self.CosmologicalExpansion = pf["CosmologicalExpansion"]
        self.PlaneParallelField = pf["PlaneParallelField"]
        self.C = pf["ClumpingFactor"]
        
        self.InitialRedshift = self.z0 = pf["InitialRedshift"]
        self.InitialHydrogenDensity = (data["HIDensity"][0] + data["HIIDensity"][0])
        self.InitialHeliumDensity = (data["HeIDensity"][0] + data["HeIIDensity"][0] + data["HeIIIDensity"][0])
        
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
        guesses = [self.cosm.nH0, 
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

    def EvolvePhotons(self, data, t, dt, h, lb):
        """
        This routine calls our solvers and updates 'data' -> 'newdata'
        """
        
        # Make data globally accessible
        self.data = data
        
        # Figure out which processors will solve which cells and create newdata dict
        self.solve_arr, newdata = self.control.DistributeDataAcrossProcessors(data, lb)
                          
        # Nice names for densities, ionized fractions
        self.n_H_arr = data["HIDensity"] + data["HIIDensity"]
        self.x_HI_arr = data["HIDensity"] / self.n_H_arr
        self.x_HII_arr = data["HIIDensity"] / self.n_H_arr
        
        if self.MultiSpecies:
            self.n_He_arr = data["HeIDensity"] + data["HeIIDensity"] + data["HeIIIDensity"]
            self.x_HeI_arr = data["HeIDensity"] / self.n_He_arr
            self.x_HeII_arr = data["HeIIDensity"] / self.n_He_arr
            self.x_HeIII_arr = data["HeIIIDensity"] / self.n_He_arr
        else: 
            self.n_He_arr = self.x_HeI_arr = self.x_HeII_arr = self.x_HeIII_arr = np.zeros_like(self.x_HI_arr)
                                                        
        # If we're in an expanding universe, dilute densities by (1 + z)**3
        self.z = None 
        self.dz = None   
        if self.CosmologicalExpansion: 
            self.z = self.cosm.TimeToRedshiftConverter(0., t, self.InitialRedshift)
            self.dz = dt / self.cosm.dtdz(self.z)
            
        # Compute column densities - meaning column density *between* source and cell
        self.ncol_HI = np.roll(np.cumsum(data["HIDensity"] * self.dx), 1)
        self.ncol_HeI = np.roll(np.cumsum(data["HeIDensity"] * self.dx), 1)
        self.ncol_HeII = np.roll(np.cumsum(data["HeIIDensity"] * self.dx), 1)
        self.ncol_HI[0] = self.ncol_HeI[0] = self.ncol_HeII[0] = neglible_column
        
        # Convenience arrays for column densities, absorbers, ion densities, and some others
        self.ncol_all = np.transpose(np.log10([self.ncol_HI, self.ncol_HeI, self.ncol_HeII]))
        self.nabs_all = np.transpose([data["HIDensity"], data["HeIDensity"], data["HeIIDensity"]])
        self.nion_all = np.transpose([data["HIIDensity"], data["HeIIDensity"], data["HeIIIDensity"]])
        self.mu_all = 1. / (self.X * (1. + self.x_HII_arr) + self.Y * (1. + self.x_HeII_arr + self.x_HeIII_arr) / 4.)
        self.ne_all = data["ElectronDensity"]
        self.nB_all = self.n_H_arr + self.n_He_arr + self.ne_all
        self.q_all = np.transpose([data["HIIDensity"], data["HeIIDensity"], data["HeIIIDensity"], 
            3. * k_B * self.nB_all * data["Temperature"] / 2.])
                
        # Retrieve indices used for N-D interpolation
        self.indices_all = []
        for i, col in enumerate(self.ncol_all):
            self.indices_all.append(self.coeff.Interpolate.GetIndices(col, np.log10(self.x_HII_arr[i])))
                                                
        # Compute optical depths *between* source and all cells. Do we only use this for timestep calculation?
        self.tau_all_arr = np.zeros([3, self.GridDimensions])    
        if not self.pf['TabulateIntegrals']:
            sigma0 = PhotoIonizationCrossSection(self.rs.E, species = 0)
            tmp_nHI = np.transpose(len(self.rs.E) * [self.ncol_HI])            
            self.tau_all_arr[0] = np.sum(tmp_nHI * sigma0, axis = 1)
                        
            if self.MultiSpecies:
                sigma1 = PhotoIonizationCrossSection(self.rs.E, species = 1)
                sigma2 = PhotoIonizationCrossSection(self.rs.E, species = 2)
                tmp_nHeI = np.transpose(len(self.rs.E) * [self.ncol_HeI])
                tmp_nHeII = np.transpose(len(self.rs.E) * [self.ncol_HeII])
                self.tau_all_arr[1] = np.sum(tmp_nHeI * sigma1, axis = 1)
                self.tau_all_arr[2] = np.sum(tmp_nHeII * sigma2, axis = 1)
        else:
            for i, col in enumerate(np.log10(self.ncol_HI)):
                self.tau_all_arr[0][i] = self.coeff.Interpolate.OpticalDepth(col, 0)
                
                if self.MultiSpecies:
                    self.tau_all_arr[1][i] = self.coeff.Interpolate.OpticalDepth(np.log10(self.ncol_HeI[i]), 1)
                    self.tau_all_arr[2][i] = self.coeff.Interpolate.OpticalDepth(np.log10(self.ncol_HeII[i]), 2)
                    
            self.tau_all_arr[0][0] = self.tau_all_arr[1][0] = self.tau_all_arr[2][0] = neglible_tau 

        self.tau_all = zip(*self.tau_all_arr) 
                                        
        # Print status, and update progress bar
        if rank == 0: 
            print "rt1d: %g < t < %g" % (t / self.TimeUnits, (t + dt) / self.TimeUnits)
        if rank == 0 and self.ProgressBar: 
            self.pbar = ProgressBar(widgets = widget, maxval = self.grid[-1]).start()        
                
        # SOLVE: c -> inf
        if self.InfiniteSpeedOfLight: 
            newdata, dtphot = self.EvolvePhotonsAtInfiniteSpeed(newdata, t, dt, h)
                
        # SOLVE: c = finite   
        else:
            newdata, dtphot = self.EvolvePhotonsAtFiniteSpeed(newdata, t, dt, h)
         
        ### 
        ## Tidy up a bit
        ###
        
        # If multiple processors at work, communicate data and timestep                                                                                          
        if (size > 1) and (self.ParallelizationMethod == 1):
            for key in newdata.keys(): 
                newdata[key] = MPI.COMM_WORLD.allreduce(newdata[key], newdata[key])
                
            dtphot = MPI.COMM_WORLD.allreduce(dtphot, dtphot) 
                                
        # Compute timestep for next cycle based on minimum dt required over entire grid                        
        if self.AdaptiveGlobalStep: 
            newdt = min(np.min(dtphot), 2 * dt)
            
            if self.CosmologicalExpansion and self.RedshiftRestrictedTimestep:
                newdt = min(newdt, self.cosm.dtdz(self.z) * self.MaxRedshiftStep)
            
        else: 
            newdt = dt            
        
        # Store timestep information
        newdata['dtPhoton'] = dtphot
                                
        # Load balance grid for next timestep                     
        if size > 1 and self.ParallelizationMethod == 1: 
            lb = self.control.LoadBalance(dtphot)   
        else: 
            lb = None      
            
        if rank == 0 and self.ProgressBar: 
            self.pbar.finish()    
                                                                                                                                                                                 
        return newdata, h, newdt, lb
        
    def EvolvePhotonsAtInfiniteSpeed(self, newdata, t, dt, h):
        """
        Solver for InfiniteSpeedOfLight = 1.
        """        
        
        # Set up timestep array for use on next cycle
        if self.AdaptiveGlobalStep:
            dtphot = 1. * np.zeros_like(self.grid)
        else:
            dtphot = dt
        
        # Could change with time for accreting black holes (or not yet implemented sources)
        Lbol = self.rs.BolometricLuminosity(t)
                                                
        # Loop over cells radially, solve rate equations, update values in data -> newdata
        for cell in self.grid:   
                        
            # If this cell belongs to another processor, continue
            if self.ParallelizationMethod == 1 and size > 1:
                if cell not in self.solve_arr: 
                    continue
                    
            # Update progressbar        
            if rank == 0 and self.ProgressBar: 
                self.pbar.update(cell * size)
                                                            
            # Read in densities for this cell
            n_e = self.data["ElectronDensity"][cell]
            n_HI = self.data["HIDensity"][cell]
            n_HII = self.data["HIIDensity"][cell]
            n_HeI = self.data["HeIDensity"][cell]
            n_HeII = self.data["HeIIDensity"][cell]
            n_HeIII = self.data["HeIIIDensity"][cell] 
                    
            # Read in ionized fractions for this cell
            x_HI = self.x_HI_arr[cell]
            x_HII = self.x_HII_arr[cell]
            x_HeI = self.x_HeI_arr[cell]
            x_HeII = self.x_HeII_arr[cell]
            x_HeIII = self.x_HeIII_arr[cell]
                                    
            # Convenience arrays for column, absorber, and ion densities plus a few others
            ncol = self.ncol_all[cell]   # actually log10(ncol)
            nabs = self.nabs_all[cell]
            nion = self.nion_all[cell]
            n_H = n_HI + n_HII
            n_He = n_HeI + n_HeII + n_HeIII                
            n_B = n_H + n_He + n_e
                                              
            # Read in temperature and internal energy for this cell
            T = self.data["Temperature"][cell]
                                                                                          
            # Read radius
            r = self.r[cell]
                            
            # Retrieve indices used for 3D interpolation
            indices = self.indices_all[cell]
            
            # Retrieve optical depth to this cell
            tau = self.tau_all[cell]
                            
            # Retrieve path length through this cell
            dx = self.dx[cell]     
                                                                         
            # Retrieve coefficients and what not.
            args = [nabs, nion, n_H, n_He, n_e]
            args.extend(self.coeff.ConstructArgs(args, indices, Lbol, r, ncol, T, dx, t, self.z))
            
            # Unpack so we have everything by name
            nabs, nion, n_H, n_He, n_e, Gamma, gamma, Beta, alpha, \
                k_H, zeta, eta, psi, xi, omega, hubble, compton = args                                                                           
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
            ######################################
            ######## Solve Rate Equations ########
            ######################################        
                                                          
            tarr, qnew, h, odeitr, rootitr = self.solver.integrate(self.RateEquations, 
                self.q_all[cell], t, t + dt, None, h, *args)
                                                                                                                                       
            # Unpack results of coupled equations
            newHII, newHeII, newHeIII, newE = qnew 
            
            if self.CosmologicalExpansion:
                n_H = self.cosm.nH0 * (1. + (self.z - self.dz))**3
                n_He = self.cosm.nHe0 * (1. + (self.z - self.dz))**3  
                n_e = newHII + newHeII + 2.0 * newHeIII
                n_B = n_H + n_He + n_e             
                                
            # Calculate neutral fractions
            newHI = n_H - newHII
            newHeI = n_He - newHeII - newHeIII 
                                                
            # Convert from internal energy back to temperature
            newT = newE * 2. / 3. / k_B / n_B
                                    
            # Store data
            newdata = self.StoreData(newdata, cell, newHI, newHII, newHeI, newHeII, newHeIII, newT,
                tau, odeitr, h, rootitr, Gamma, k_H, Beta, alpha, zeta, eta, psi, gamma, xi, 
                omega, hubble, compton)
                                                                                                            
            ######################################
            ################ DONE ################
            ######################################
                
            # Adjust timestep based on maximum allowed neutral fraction change     
            if self.AdaptiveGlobalStep:
                dtphot[cell] = self.control.ComputePhotonTimestep(tau, 
                    nabs, nion, ncol, n_H, n_He, n_e, n_B, Gamma, gamma, Beta, alpha, k_H, 
                        zeta, eta, psi, xi, omega, hubble, compton, newT, self.z, dt) 
        
        return newdata, dtphot
        
    def EvolvePhotonsAtFiniteSpeed(self, newdata, t, dt, h):
        """
        Solver for InfiniteSpeedOfLight = 0.
        
        PhotonPackage guide: 
            pack = [EmissionTime, EmissionTimeInterval, ncolHI, ncolHeI, ncolHeII, Energy]
        """ 
        
        # Set up timestep array for use on next cycle
        if self.AdaptiveGlobalStep:
            dtphot = 1.e50 * np.ones_like(self.grid)
        else:
            dtphot = dt
            
        # Could change with time for accreting black holes
        Lbol = self.rs.BolometricLuminosity(t)  
            
        # Photon packages going from oldest to youngest - will have to create it on first timestep
        if t == 0: 
            packs = []
        else: 
            packs = list(self.data['PhotonPackages']) 
        
        if packs and self.OnePhotonPackagePerCell:
            t_packs = np.array(zip(*packs)[0])    # Birth times for all packages
            r_packs = (t - t_packs) * c
            
            # If there are already photon packages in the first cell, add energy in would-be packet to preexisting one
            if r_packs[-1] < (self.StartRadius * self.LengthUnits):
                packs[-1][-1] += Lbol * dt
            # Launch new photon package
            else:
                packs.append(np.array([t, dt, neglible_column, neglible_column, neglible_column, Lbol * dt]))
        else: 
            packs.append(np.array([t, dt, neglible_column, neglible_column, neglible_column, Lbol * dt]))
            
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
                
                # These quantities will be different (in general) for each step
                # of the while loop
                n_e = newdata["ElectronDensity"][cell]
                n_HI = newdata["HIDensity"][cell]
                n_HII = newdata["HIIDensity"][cell]
                n_HeI = newdata["HeIDensity"][cell]
                n_HeII = newdata["HeIIDensity"][cell]
                n_HeIII = newdata["HeIIIDensity"][cell] 
                n_H = n_HI + n_HII
                n_He = n_HeI + n_HeII + n_HeIII
                
                # Read in ionized fractions for this cell
                x_HI = n_HI / n_H
                x_HII = n_HII / n_H
                x_HeI = n_HeI / n_He
                x_HeII = n_HeII = n_He
                x_HeIII = n_HeIII = n_He
                
                # Compute mean molecular weight for this cell
                mu = 1. / (self.X * (1. + x_HII) + self.Y * (1. + x_HeII + x_HeIII) / 4.)
                
                # Retrieve path length through this cell
                dx = self.dx[cell]     
                                        
                # Crossing time
                dct = self.CellCrossingTime[cell]                        
                                        
                # For convenience     
                nabs = np.array([n_HI, n_HeI, n_HeII])
                nion = np.array([n_HII, n_HeII, n_HeIII])
                n_H = n_HI + n_HII
                n_He = n_HeI + n_HeII + n_HeIII
                n_B = n_H + n_He + n_e
                
                # Compute internal energy for this cell
                T = newdata["Temperature"][cell]
                E = 3. * k_B * T * n_B / mu / 2.
                
                q_cell = [n_HII, n_HeII, n_HeIII, E]
                                
                # Add columns of this cell
                packs[j][2] += newdata['HIDensity'][cell] * dc * dct * c  
                packs[j][3] += newdata['HeIDensity'][cell] * dc * dct * c 
                packs[j][4] += newdata['HeIIDensity'][cell] * dc * dct * c
                ncol = np.log10(packs[j][2:5])
                
                ######################################
                ######## Solve Rate Equations ########
                ######################################
                
                # Retrieve indices used for 3D interpolation
                indices = None
                if self.MultiSpecies > 0: 
                    indices = self.coeff.Interpolate.GetIndices3D(ncol)
                
                # Retrieve coefficients and what not.
                args = [nabs, nion, n_H, n_He, n_e]
                args.extend(self.coeff.ConstructArgs(args, indices, Lbol, r, ncol, T, dx, t))
                
                # Unpack so we have everything by name
                nabs, nion, n_H, n_He, n_e, Gamma, gamma, Beta, alpha, k_H, zeta, eta, psi, xi, omega, theta = args
                                                                                                         
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
                         
                # Store data
                newdata = self.StoreData(newdata, cell, newHI, newHII, newHeI, newHeII, newHeIII, newT,
                    self.tau_all[cell], odeitr, h, rootitr, Gamma, k_H, Beta, alpha, zeta, eta, psi, gamma, xi, omega)
                                                         
                cell_pack += dc
                                                
                ######################################
                ################ DONE ################     
                ######################################                                   
            
        # Adjust timestep based on maximum allowed neutral fraction change     
        if self.AdaptiveGlobalStep:
            for cell in self.grid:
                Gamma = [newdata['PhotoIonizationRate%i' % i] for i in np.arange(3)]
                gamma = [newdata['SecondaryIonizationRate%i' % i] for i in np.arange(3)]
                Beta = [newdata['CollisionalIonizationRate%i' % i] for i in np.arange(3)]
                alpha = [newdata['RadiativeRecombinationRate%i' % i] for i in np.arange(3)]
                xi = [newdata['DielectricRecombinationRate%i' % i] for i in np.arange(3)]
                newT = newdata["Temperature"][cell]
                
                dtphot[cell] = self.control.ComputePhotonTimestep(self.tau_all_arr[:,cell], 
                    self.nabs_all[cell], self.nion_all[cell], self.ncol_all[cell], self.n_H_arr[cell], self.n_He_arr[cell], 
                    self.ne_all[cell], self.nB_all[cell], 
                    Gamma, gamma, Beta, alpha, k_H, zeta, eta, psi, xi, omega, theta, newT, dt) 
                    
                if self.LightCrossingTimeRestrictedTimestep: 
                    dtphot[cell] = min(dtphot[cell], self.LightCrossingTimeRestrictedTimestep * self.CellCrossingTime[cell])    
                    
        # Update photon packages        
        if not self.InfiniteSpeedOfLight: 
            newdata['PhotonPackages'] = self.UpdatePhotonPackages(packs, t + dt)
        
        return newdata, dtphot
    
    def UpdatePhotonPackages(self, packs, t_next):
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
                  
        return np.array(packs)
       
    def StoreData(self, newdata, cell, newHI, newHII, newHeI, newHeII, newHeIII, newT,
        tau, odeitr, h, rootitr, Gamma, k_H, Beta, alpha, zeta, eta, psi, gamma, xi, 
        omega, hubble, compton):
        """
        Copy fields to newdata dictionary.
        """
        
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
                    
        return newdata            
        
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
        hubble = args[15]
        compton = args[16]
                        
        dqdt = self.zeros_tmp
        
        # Neutrals (current time-step)
        nHI = n_H - q[0]
        nHeI = n_He - q[1] - q[2]
        nHeII = q[1]
                
        # Always solve hydrogen rate equation
        dqdt[0] = (Gamma[0] + Beta[0] * n_e) * nHI + \
                  (gamma[0][0] * nHI + gamma[0][1] * nHeI + gamma[0][2] * nHeII) - \
                   alpha[0] * self.C * n_e * q[0]

        if self.CosmologicalExpansion:
            dqdt[0] -= 3. * q[0] * hubble

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
                + np.sum(eta * nion) + np.sum(psi * nabs) + q[2] * omega[1])      
                               
            if self.CosmologicalExpansion:
                dqdt[3] = -4. * hubble * q[3] / 3.
                
                if self.ComptonHeating:
                    dqdt[3] += compton
        
        return dqdt
        
