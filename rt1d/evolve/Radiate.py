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
from .SpinTemperature import SpinTemperature
from .Hydrogen import Hydrogen

from scipy.integrate import ode

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

E_th = [13.6, 24.6, 54.4]

neglible_tau = 1e-12
neglible_column = 10

class Radiate:
    def __init__(self, pf, rs, g):
        self.pf = pf
        
        # Classes
        self.rs = rs
        self.cosm = Cosmology(pf)
        self.control = ControlSimulation(pf)
        self.coeff = RateCoefficients(pf, rs, g)
        self.Ts = SpinTemperature(pf)   
        self.HI = Hydrogen(pf)  
        
        self.ProgressBar = pf["ProgressBar"] and pb
        
        # Grid/units/etc
        self.GridDimensions = int(pf["GridDimensions"])
        self.grid = np.arange(self.GridDimensions)
        self.StopTime = pf['StopTime'] * self.pf['TimeUnits']
        self.StartCell = pf['StartRadius'] * self.GridDimensions
        self.R0 = pf['StartRadius'] * pf['LengthUnits']
        
        # Time-stepping
        self.AdaptiveGlobalStep = bool(self.pf['HIRestrictedTimestep']) or \
            bool(self.pf['ElectronRestrictedTimestep'])
        if self.pf['MultiSpecies']:
            self.AdaptiveGlobalStep |= (bool(self.pf['HeIRestrictedTimestep']) or \
                bool(self.pf['HeIIRestrictedTimestep']) or bool(self.pf['HeIIIRestrictedTimestep']))
            
        # Deal with log-grid, compute dx
        if pf['LogarithmicGrid']:
            self.r = np.logspace(np.log10(self.R0), \
                np.log10(pf['LengthUnits']), self.GridDimensions + 1)
        else:
            self.r = np.linspace(self.R0, pf['LengthUnits'], self.GridDimensions + 1)
        
        self.dx = np.diff(self.r)   
        self.r = self.r[0:-1] 
        self.CellCrossingTime = self.dx / c
                
        # For convenience 
        self.zeros_tmp = np.zeros(4)
        
        # Initialize solver
        self.solver = ode(self.RateEquations).set_integrator('vode', 
            method = 'bdf', nsteps = 1e4, with_jacobian = True,
            atol = 1e-12, rtol = 1e-8)
        
    def EvolvePhotons(self, data, t, dt, h, lb):
        """
        This routine calls our solvers and updates 'data' -> 'newdata'
        """
        
        # Make data globally accessible
        self.data = data
        
        # Figure out which processors will solve which cells and create newdata dict
        self.solve_arr, newdata = self.control.DistributeDataAcrossProcessors(data, lb)
                          
        # If we're in an expanding universe, prepare to dilute densities by (1 + z)**3
        self.z = 0 
        self.dz = 0   
        if self.pf['CosmologicalExpansion']: 
            self.z = self.cosm.TimeToRedshiftConverter(0., t, self.pf['InitialRedshift'])
            self.dz = dt / self.cosm.dtdz(self.z)
               
        # Nice names for densities, ionized fractions
        self.n_H_arr = data["HIDensity"] + data["HIIDensity"]
        self.x_HI_arr = data["HIDensity"] / self.n_H_arr
        self.x_HII_arr = data["HIIDensity"] / self.n_H_arr
        
        if self.pf['MultiSpecies']:
            self.n_He_arr = data["HeIDensity"] + data["HeIIDensity"] + data["HeIIIDensity"]
            self.x_HeI_arr = data["HeIDensity"] / self.n_He_arr
            self.x_HeII_arr = data["HeIIDensity"] / self.n_He_arr
            self.x_HeIII_arr = data["HeIIIDensity"] / self.n_He_arr
        else: 
            self.n_He_arr = self.x_HeI_arr = self.x_HeII_arr = self.x_HeIII_arr = np.zeros_like(self.x_HI_arr)
                                                        
        # Compute column densities - meaning column density *between* source and cell
        self.ncol_HI = np.roll(np.cumsum(data["HIDensity"] * self.dx), 1)
        self.ncol_HeI = np.roll(np.cumsum(data["HeIDensity"] * self.dx), 1)
        self.ncol_HeII = np.roll(np.cumsum(data["HeIIDensity"] * self.dx), 1)
        self.ncol_HI[0] = self.ncol_HeI[0] = self.ncol_HeII[0] = neglible_column
        
        # Convenience arrays for column densities, absorbers, ion densities, and some others
        self.ncol_all = np.transpose(np.log10([self.ncol_HI, self.ncol_HeI, self.ncol_HeII]))
        self.nabs_all = np.transpose([data["HIDensity"], data["HeIDensity"], data["HeIIDensity"]])
        self.nion_all = np.transpose([data["HIIDensity"], data["HeIIDensity"], data["HeIIIDensity"]])
        self.mu_all = 1. / (self.cosm.X * (1. + self.x_HII_arr) + self.cosm.Y * (1. + self.x_HeII_arr + self.x_HeIII_arr) / 4.)
        self.ne_all = data["ElectronDensity"]
        self.nB_all = self.n_H_arr + self.n_He_arr + self.ne_all
        
        self.q_all = np.transpose([self.x_HII_arr, self.x_HeII_arr, self.x_HeIII_arr, 
            3. * k_B * self.nB_all * data["Temperature"] / 2.])
                                                                                                     
        # Retrieve indices used for N-D interpolation
        self.indices_all = []
        for i, col in enumerate(self.ncol_all):
            tmp = []
            for rs in self.rs.all_sources:
                if rs.TableAvailable:
                    tmp.append(rs.Interpolate.GetIndices([col[0], col[1], col[2], np.log10(self.x_HII_arr[i]), t]))
                else:
                    tmp.append(None)
            self.indices_all.append(tmp)
                                                
        # Compute tau *between* source and all cells
        tau_all_arr = self.ComputeOpticalDepths([self.ncol_HI, self.ncol_HeI, self.ncol_HeII])
        self.tau_all = zip(*tau_all_arr) 
                                        
        # Print status, and update progress bar
        if rank == 0: 
            print "rt1d: %g < t < %g" % (t / self.pf['TimeUnits'], (t + dt) / self.pf['TimeUnits'])
        if rank == 0 and self.ProgressBar: 
            self.pbar = ProgressBar(widgets = widget, maxval = self.grid[-1]).start()        
                
        # SOLVE: c -> inf
        if self.pf['InfiniteSpeedOfLight']: 
            newdata, dtphot = self.EvolvePhotonsAtInfiniteSpeed(newdata, t, dt, h)
                
        # SOLVE: c = finite   
        else:
            newdata, dtphot = self.EvolvePhotonsAtFiniteSpeed(newdata, t, dt, h)
              
        ### 
        ## Tidy up a bit
        ###
        
        # If multiple processors at work, communicate data and timestep                                                                                          
        if (size > 1) and (self.pf['ParallelizationMethod'] == 1):
            for key in newdata.keys(): 
                newdata[key] = MPI.COMM_WORLD.allreduce(newdata[key], newdata[key])
                
            dtphot = MPI.COMM_WORLD.allreduce(dtphot, dtphot) 
                                
        # Compute timestep for next cycle based on minimum dt required over entire grid                        
        if self.AdaptiveGlobalStep: 
            newdt = min(np.min(dtphot), 2 * dt)
            
            if self.pf['CosmologicalExpansion'] and self.pf['RedshiftRestrictedTimestep']:
                newdt = min(newdt, self.cosm.dtdz(self.z) * self.pf['MaxRedshiftStep'])
            
        else: 
            newdt = dt            
        
        # Store timestep information
        newdata['dtPhoton'] = dtphot
                                
        # Load balance grid for next timestep                     
        if size > 1 and self.pf['ParallelizationMethod'] == 1: 
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
        Lbol = []
        for rs in self.rs.all_sources:
            Lbol.append(rs.BolometricLuminosity(t))
                                                
        # Loop over cells radially, solve rate equations, update values in data -> newdata
        for cell in self.grid:   
                        
            # If this cell belongs to another processor, continue
            if self.pf['ParallelizationMethod'] == 1 and size > 1:
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
            args.extend(self.coeff.ConstructArgs(args, indices, Lbol, r, ncol, T, dx, t, self.z, cell))
            
            # Unpack so we have everything by name
            nabs, nion, n_H, n_He, n_e, Gamma, gamma, Beta, alpha, \
                k_H, zeta, eta, psi, xi, omega, hubble, compton, Jc, Ji = args     
                
            # Sum ionization + heating over all sources (last axis)            
            args[5] = np.sum(Gamma, axis = -1) 
            args[6] = np.sum(gamma, axis = -1)      
            args[9] = np.sum(k_H, axis = -1)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
            ######################################
            ######## Solve Rate Equations ########
            ###################################### 
                                                                                                                                    
            self.solver.set_initial_value(self.q_all[cell], t).set_f_params(args).set_jac_params(args)
            self.solver.integrate(t + dt)
            newxHII, newxHeII, newxHeIII, newE = self.solver.y 
            newxHII, newxHeII, newxHeIII, newE = self.ConserveParticleNumber(newxHII, newxHeII, newxHeIII, newE) 
                                            
            # Convert from internal energy back to temperature
            if self.pf['Isothermal']:
                newT = T
            else:
                newT = newE * 2. / 3. / k_B / n_B                                
                                            
            if self.pf['CosmologicalExpansion']:
                n_H = self.cosm.nH0 * (1. + (self.z - self.dz))**3
                n_He = self.cosm.nHe0 * (1. + (self.z - self.dz))**3  
                n_e = newxHII * n_H + newxHeII * n_He + 2.0 * newxHeIII * n_He
                n_B = n_H + n_He + n_e
                 
            # Calculate neutral fractions
            newHI = n_H * (1. - newxHII)
            newHII = n_H * newxHII
            newHeI = n_He * (1. - newxHeII - newxHeIII)
            newHeII = n_He * newxHeII
            newHeIII = n_He * newxHeIII
            
            # Derive Ts, dTb
            delta = 0
            Ts = self.Ts.Ts(self.z, n_H, n_e, newT, newxHII, np.sum(Jc), np.sum(Ji))
            dTb = self.HI.BrightnessTemperature(self.z, newxHII, delta, Ts)
                                                                        
            # Store data
            newdata = self.StoreData(newdata, cell, newHI, newHII, newHeI, newHeII, newHeIII, newT,
                tau, Gamma, k_H, Beta, alpha, zeta, eta, psi, gamma, xi, omega, hubble, compton, 
                Jc, Ji, Ts, dTb)
                                                                                                            
            ######################################
            ################ DONE ################
            ######################################
                
            # Adjust timestep based on maximum allowed neutral fraction change     
            if self.AdaptiveGlobalStep:
                dtphot[cell] = self.control.ComputePhotonTimestep(tau, 
                    [newHI, newHeI, newHeII], [newHII, newHeII, newHeIII], 
                    ncol, n_H, n_He, n_e, n_B, Gamma, gamma, Beta, alpha, k_H, 
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
            
        Lbol = self.rs.BolometricLuminosity(t)    
            
        # Photon packages going from oldest to youngest - will have to create it on first timestep
        if t == 0: 
            packs = []
        else: 
            packs = list(self.data['PhotonPackages']) 
        
        # Add one for this timestep
        packs.append(np.array([t, dt, neglible_column, neglible_column, neglible_column, Lbol * dt]))
            
        # Loop over photon packages, updating values in cells: data -> newdata
        for j, pack in enumerate(packs):
            t_birth = pack[0]
            r_pack = (t - t_birth) * c        # Position of package before evolving photons
            r_max = r_pack + dt * c           # Furthest this package will get this timestep
                                                                             
            # Cells we need to know about - not necessarily integer
            cell_pack = (r_pack - self.R0) * self.GridDimensions / self.pf['LengthUnits']
            cell_pack_max = (r_max - self.R0) * self.GridDimensions / self.pf['LengthUnits'] - 1
            cell_t = t  
                      
            Lbol = pack[-1] / pack[1]          
                                            
            # Advance this photon package as far as it will go on this global timestep  
            while cell_pack < cell_pack_max:
                                                        
                # What cell are we in
                if cell_pack < 0:
                    cell = -1
                else:    
                    cell = int(cell_pack)
                
                if cell >= self.GridDimensions: 
                    break
                
                # Compute dc (like dx but in fractional cell units)
                # Really how far this photon can go in this step
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
                   
                # If photons haven't hit first cell interface yet, evolve in time                
                if cell < 0:
                    cell_pack += dc
                    cell_t += subdt
                    continue        
                    
                # Current radius in code units                                                                                                                                                                                                                                                                                                                          
                r = cell_pack * self.pf['LengthUnits'] / self.pf['GridDimensions']
                
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
                mu = 1. / (self.cosm.X * (1. + x_HII) + self.cosm.Y * (1. + x_HeII + x_HeIII) / 4.)
                
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
                packs[j][2] += newdata['HIDensity'][cell] * dc * dx
                packs[j][3] += newdata['HeIDensity'][cell] * dc * dx 
                packs[j][4] += newdata['HeIIDensity'][cell] * dc * dx
                ncol = np.log10(packs[j][2:5])
                
                ######################################
                ######## Solve Rate Equations ########
                ######################################
                
                # Retrieve indices used for interpolation
                indices = self.coeff.Interpolate.GetIndices(ncol)
                
                # Retrieve coefficients and what not.
                args = [nabs, nion, n_H, n_He, n_e]                
                args.extend(self.coeff.ConstructArgs(args, indices, Lbol, r, ncol, T, dx * dc, t, self.z, cell))
           
                # Unpack so we have everything by name
                nabs, nion, n_H, n_He, n_e, Gamma, gamma, Beta, alpha, \
                    k_H, zeta, eta, psi, xi, omega, hubble, compton = args        
                                                                                                         
                ######################################
                ######## Solve Rate Equations ########
                ######################################                          
                                
                tarr, qnew, h, odeitr, rootitr = self.solver.integrate(self.RateEquations, 
                    q_cell, cell_t, cell_t + subdt, self.z, self.z - self.dz, None, h, *args)                 
                                    
                # Unpack results of coupled equations - Remember: these are lists and we only need the last entry 
                newHII, newHeII, newHeIII, newE = qnew    
                
                # Weight by volume if less than a cell traversed
                if dc < 1:
                    dnHII = newHII - newdata['HIIDensity'][cell]
                    dnHeII = newHeII - newdata['HeIIDensity'][cell]
                    dnHeIII = newHeIII - newdata['HeIIIDensity'][cell]
                    dV = self.coeff.ShellVolume(r, dx * dc) / self.coeff.ShellVolume(self.r[cell], dx)
                    newHII = newdata['HIIDensity'][cell] + dnHII * dV 
                    newHeII = newdata['HeIIDensity'][cell] + dnHeII * dV
                    newHeIII = newdata['HeIIIDensity'][cell] + dnHeIII * dV
                                
                # Calculate neutral fractions
                newHI = n_H - newHII
                newHeI = n_He - newHeII - newHeIII               
                
                # Convert from internal energy back to temperature
                newT = newE * 2. * mu / 3. / k_B / n_B     
                                 
                # Store data
                newdata = self.StoreData(newdata, cell, newHI, newHII, newHeI, newHeII, newHeIII, newT,
                    self.tau_all[cell], odeitr, h, rootitr, Gamma, k_H, Beta, alpha, zeta, eta, psi, gamma, xi, 
                    omega, hubble, compton)                    
                                                         
                cell_pack += dc
                cell_t += subdt
                                                
                ######################################
                ################ DONE ################     
                ######################################  
                
        # Adjust timestep for next cycle
        if self.AdaptiveGlobalStep:
            n_HI = newdata['HIDensity']
            n_HII = newdata['HIIDensity']
            n_H_all = n_HI + n_HII
            n_HeI = newdata['HeIDensity']
            n_HeII = newdata['HeIIDensity']
            n_HeIII = newdata['HeIIIDensity']
            n_He_all = n_HeI + n_HeII + n_HeIII
            n_e_all = n_HII + n_HeII + 2 * n_HeIII
            T = newdata['Temperature']
            n_B_all = n_H_all + n_He_all + n_e_all
            
            ncol_HI = np.roll(np.cumsum(n_HI * self.dx), 1)
            ncol_HeI = np.roll(np.cumsum(n_HeI * self.dx), 1)
            ncol_HeII = np.roll(np.cumsum(n_HeII * self.dx), 1)
            ncol_HI[0] = ncol_HeI[0] = ncol_HeII[0] = neglible_column
            ncol = np.transpose(np.log10([ncol_HI, ncol_HeI, ncol_HeII]))   
            
            tau = self.ComputeOpticalDepths([ncol_HI, ncol_HeI, ncol_HeII])
            
            for cell in self.grid:
                r = self.r[cell]
                dx = self.dx[cell]
                nabs = np.array([n_HI[cell], n_HeI[cell], n_HeII[cell]])
                nion = np.array([n_HII[cell], n_HeII[cell], n_HeIII[cell]])
                
                indices = self.coeff.Interpolate.GetIndices(ncol[cell])
                               
                args = [nabs, nion, n_H_all[cell], n_He_all[cell], n_e_all[cell]]  
                args.extend(self.coeff.ConstructArgs(args, indices, Lbol, r, ncol[cell], T[cell], dx, t, self.z))
           
                # Unpack so we have everything by name
                nabs, nion, n_H, n_He, n_e, Gamma, gamma, Beta, alpha, \
                    k_H, zeta, eta, psi, xi, omega, hubble, compton = args 
                
                dtphot[cell] = self.control.ComputePhotonTimestep(tau[:,cell], 
                    nabs, nion, ncol[cell], n_H, n_He, 
                    n_e, n_B_all[cell], Gamma, gamma, Beta, alpha, k_H, zeta, 
                    eta, psi, xi, omega, hubble, compton, T[cell], self.z, dt) 
                    
                if self.pf['LightCrossingTimeRestrictedTimestep']: 
                    dtphot[cell] = min(dtphot[cell], 
                        self.LightCrossingTimeRestrictedTimestep * self.CellCrossingTime[cell])    
                    
        # Update photon packages   
        newdata['PhotonPackages'] = np.array(self.UpdatePhotonPackages(packs, t + dt))
                
        return newdata, dtphot
        
    def ConserveParticleNumber(self, newxHII, newxHeII, newxHeIII, newE):
        """
        Take solver.y and make sure ion fractions sum to 1 - MinimumSpeciesFraction.
        """    
        
        if newxHII >= 1:
            newxHII = 1. - self.pf['MinimumSpeciesFraction']
            
        if (newxHeII + newxHeIII) >= 1:
            if newxHeII > newxHeIII:
                newxHeII += 1. - (newxHeII + newxHeIII)
                newxHeIII -= self.pf['MinimumSpeciesFraction']
            else:
                newxHeIII += 1. - (newxHeII + newxHeIII)
                newxHeII -= self.pf['MinimumSpeciesFraction']
    
        return newxHII, newxHeII, newxHeIII, newE
    
    def UpdatePhotonPackages(self, packs, t_next):
        """
        Get rid of old photon packages.
        """    
                
        to_eliminate = []
        for i, pack in enumerate(packs):
            if (t_next - pack[0]) > (self.pf['LengthUnits'] / c): 
                to_eliminate.append(i)
            
        to_eliminate.reverse()    
        for element in to_eliminate: 
            packs.pop(element)
                  
        return np.array(packs)
       
    def StoreData(self, newdata, cell, newHI, newHII, newHeI, newHeII, newHeIII, newT,
        tau, Gamma, k_H, Beta, alpha, zeta, eta, psi, gamma, xi, omega, hubble, compton, 
        Jc, Ji, Ts, dTb, packs = None):
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
        newdata["SpinTemperature"][cell] = Ts
        newdata["BrightnessTemperature"][cell] = dTb
                
        if self.pf['OutputRates']:
            for i in xrange(3):
                
                for j in xrange(self.rs.Ns):
                    newdata['PhotoIonizationRate%i_src%i' % (i, j)][cell] = Gamma[i][j]
                    newdata['PhotoHeatingRate%i_src%i' % (i, j)][cell] = k_H[i][j]
                    newdata['SecondaryIonizationRate%i_src%i' % (i, j)][cell] = gamma[i,:,j] 
                    
                    newdata['InjectedLyAFlux%i_src%i' % (i, j)][cell] = Ji[i][j]
                    
                    if i == 0:
                        newdata['ContinuumLyAFlux_src%i' % j][cell] = Jc[j]
                
                newdata['CollisionalIonizationRate%i' % i][cell] = Beta[i] 
                newdata['RadiativeRecombinationRate%i' % i][cell] = alpha[i] 
                newdata['CollisionalIonizationCoolingRate%i' % i][cell] = zeta[i] 
                newdata['RecombinationCoolingRate%i' % i][cell] = eta[i] 
                newdata['CollisionalExcitationCoolingRate%i' % i][cell] = psi[i]                
                
                if i == 2:
                    newdata['DielectricRecombinationRate'][cell] = xi[i]
                    newdata['DielectricRecombinationCoolingRate'][cell] = omega[i]   
                    
            newdata['HubbleCoolingRate'][cell] = hubble                      
                    
        return newdata            
        
    def RateEquations(self, t, q, args):    
        """
        This function returns the right-hand side of our ODE's (Equations 1, 2, 3 and 9 in Mirocha et al. 2012).

        q = [x_HII, x_HeII, x_HeIII, E] - our four coupled equations. q = generalized quantity I guess.
        
        args = (nabs, nion, n_H, n_He, n_e, Gamma, gamma, Beta, alpha, k_H, zeta, eta, psi, xi, omega)
        
        where nabs, nion, Gamma, gamma, Beta, alpha, k_H, zeta, eta, psi, xi = 3 element arrays 
            (one entry per species)
                    
        """
        
        nabs, nion, n_H, n_He, n_e, Gamma, gamma, Beta, alpha, k_H, zeta, eta, \
            psi, xi, omega, hubble, compton, Jc, Ji = args
                        
        dqdt = self.zeros_tmp
        
        # Neutrals (current time-step)
        nHI = n_H * (1. - q[0])
        nHII = n_H * q[0]
        nHeI = n_He * (1. - q[1] - q[2])
        nHeII = n_He * q[1]
        nHeIII = n_He * q[2]
        
        nabs = [nHI, nHeI, nHeII]
        nion = [nHII, nHeII, nHeIII]
        
        xHI = 1. - q[0]
        xHII = q[0]
        xHeI = 1. - q[1] - q[2]
        xHeII = q[1]
        xHeIII = q[2]
        
        n_e = q[0] * n_H + q[1] * n_He + 2. * q[2] * n_He
        
        # Always solve hydrogen rate equation
        dqdt[0] = (Gamma[0] + Beta[0] * n_e) * xHI + \
                  (gamma[0][0] * xHI + gamma[0][1] * nHeI / n_H + gamma[0][2] * nHeII / n_H) - \
                   alpha[0] * n_e * xHII        
                
        # Helium rate equations  
        if self.pf['MultiSpecies']:       
            dqdt[1] = (Gamma[1] + Beta[1] * n_e) * xHeI + \
                      (gamma[1][0] * nHI / n_He + gamma[1][1] * nHeI / n_He + gamma[1][2] * xHeII / n_He)  + \
                       alpha[2] * n_e * xHeIII - \
                      (Beta[1] + alpha[1] + xi[1]) * n_e * nHeII / n_He
                              
            dqdt[2] = (Gamma[2] + Beta[2] * n_e) * xHeII - alpha[2] * n_e * nHeIII / n_He
                
        # Temperature evolution - looks dumb but using np.sum is slow
        if not self.pf['Isothermal']:
            phoheat = k_H[0] * nHI
            ioncool = zeta[0] * nHI
            reccool = eta[0] * nHII
            exccool = psi[0] * nHI
            
            if self.pf['MultiSpecies']:
                phoheat += k_H[1] * nabs[1] + k_H[2] * nabs[2]
                ioncool += zeta[1] * nabs[1] + zeta[2] * nabs[2]
                reccool += eta[1] * nion[1] + eta[2] * nion[2]
                exccool += psi[1] * nabs[1] + psi[2] * nabs[2]
            
            dqdt[3] = phoheat - n_e * (ioncool + reccool + exccool + nHeIII * omega[1])
                               
            if self.pf['CosmologicalExpansion']:
                dqdt[3] -= 2. * hubble * q[3]
               
                #if self.pf['ComptonHeating']:
                #    dqdt[3] += compton
            
        return dqdt
        
    def Jacobian(self, t, q, args):
        """
        Jacobian of the rate equations.
        """    
        
        nabs, nion, n_H, n_He, n_e, Gamma, gamma, Beta, alpha, k_H, zeta, eta, \
            psi, xi, omega, hubble, compton = args
                                
        # Neutrals (current time-step)
        nHI = n_H * (1. - q[0])
        nHII = n_H * q[0]
        nHeI = n_He * (1. - q[1] - q[2])
        nHeII = n_He * q[1]
        nHeIII = n_He * q[2]
        
        xHI = 1. - q[0]
        xHII = q[0]
        xHeI = 1. - q[1] - q[2]
        xHeII = q[1]
        xHeIII = q[2]
        
        J = np.zeros([4, 4])
        
        J[0][0] = -(Gamma[0] + Beta[0] * n_e - gamma[0][0]) - alpha[0] * n_e
        J[0][1] = -gamma[0][1] / n_H + gamma[0][2] / n_H
        J[0][2] = -gamma[0][1] / n_H
        J[1][0] = -gamma[1][0] / n_He
        J[1][1] = -(Gamma[1] + Beta[0] * n_e + gamma[1][1] / n_He - gamma[1][2] / n_He) \
                  +(Beta[1] + alpha[1] + xi[1]) * n_e / n_He
        J[1][2] = -gamma[1][1] / n_He + alpha[2] * n_e
        J[2][1] = (Gamma[2] + Beta[2] * n_e)
        J[2][2] = -alpha[2] * n_e / n_He
        J[3][0] = n_e * (zeta[0] - eta[0] + psi[0]) - k_H[0]
        J[3][1] = n_e * (zeta[1] - eta[1] + psi[1]) - k_H[1]
        J[3][2] = n_e * (zeta[2] - eta[2] + psi[2] - omega[1]) - k_H[2]
                                                               
        return J
        
    def ComputeOpticalDepths(self, ncol):
        """
        Compute optical depths *between* source and all cells.  Used solely (I think)
        for calculating next timestep. 
        """
        
        tau_all_arr = np.zeros([3, self.GridDimensions, self.pf['NumberOfSources']]) 
        for rs, source in enumerate(self.rs.all_sources):
            if not source.TableAvailable:
                tmp_nHI = np.transpose(len(source.E) * [ncol[0]])   
                tau_all_arr[0,:,rs] = np.sum(tmp_nHI * source.sigma[0, rs], axis = 1)
                            
                if self.pf['MultiSpecies']:
                    tmp_nHeI = np.transpose(len(source.E) * [ncol[1]])
                    tmp_nHeII = np.transpose(len(source.E) * [ncol[2]])
                    tau_all_arr[1,:,rs] = np.sum(tmp_nHeI * source.sigma[1, rs], axis = 1)
                    tau_all_arr[2,:,rs] = np.sum(tmp_nHeII * source.sigma[2, rs], axis = 1)
            else:
                for i, col in enumerate(np.log10(ncol[0])):
                    tau_all_arr[0,i,rs] = source.Interpolate.OpticalDepth(col, 0)
                    
                    if self.pf['MultiSpecies']:
                        tau_all_arr[1,i,rs] = source.Interpolate.OpticalDepth(np.log10(ncol[1][i]), 1)
                        tau_all_arr[2,i,rs] = source.Interpolate.OpticalDepth(np.log10(ncol[2][i]), 2)
                        
                tau_all_arr[0][0][rs] = tau_all_arr[1][0] = tau_all_arr[2][0] = neglible_tau 

        # Take minimum optical depth over sources to be conservative with dt
        return np.min(tau_all_arr, axis = -1)
