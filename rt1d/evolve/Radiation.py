"""

Radiation.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Sep 21 13:03:52 2012

Description: Compute properties of the radiation field, pass to Radiate, 
which calls chemistry solver.

"""

import numpy as np
from ..util import parse_kwargs
from .Chemistry import Chemistry
from ..physics.Constants import *
from .RadiationField import RadiationField

class Radiation:
    def __init__(self, grid, sources, **kwargs):
        self.pf = parse_kwargs(**kwargs)
        self.grid = grid
        self.srcs = sources
                
        # Initialize chemistry network / solver
        self.chem = Chemistry(grid, rt=kwargs['radiative_transfer'],
            rate_src=self.pf['rate_source'],
            rtol=self.pf['solver_rtol'], atol=self.pf['solver_atol'])
        
        # Initialize RT solver
        if self.srcs is not None:
            self.rfield = RadiationField(grid, sources, **kwargs)
    
    @property
    def finite_c(self):
        if hasattr(self, 'rfield'):
            return self.rfield.finite_c
        return False        
        
    def Evolve(self, data, t, dt, z=None, **kwargs):
        """
        This routine calls our solvers and updates 'data' -> 'newdata'
        
        PhotonPackage guide: 
        pack = [EmissionTime, EmissionTimeInterval, NHI, NHeI, NHeII, E]
        
        """
                
        # Make data globally accessible
        self.data = data.copy()
                
        # Figure out which processors will solve which cells and create newdata dict
        #self.solve_arr, newdata = self.control.DistributeDataAcrossProcessors(data, lb)
        
        # Set up photon packages    
        if self.finite_c and t == 0:
            self.data['photon_packages'] = []
        
        # Compute source dependent rate coefficients
        self.kwargs = {}
        if self.pf['radiative_transfer']:
            
            if self.finite_c:
                raise NotImplementedError('Finite speed-of-light solver not implemented.')
            
            else:    
                Gamma_src, gamma_src, Heat_src, Ja_src = \
                    self.rfield.SourceDependentCoefficients(data, t, z, 
                        **kwargs)    
                
            if len(self.srcs) > 1:
                for i, src in enumerate(self.srcs):
                    self.kwargs.update({'Gamma_%i' % i: Gamma_src[i], 
                        'gamma_%i' % i: gamma_src[i],
                        'Heat_%i' % i: Heat_src[i]})
                    
                    if not self.pf['approx_lya']:
                        self.kwargs.update({'Ja_%i' % i: Ja_src[i]})
            
            Gamma = np.sum(Gamma_src, axis=0)
            gamma = np.sum(gamma_src, axis=0)
            Heat = np.sum(Heat_src, axis=0)
                             
            # Compute Lyman-Alpha emission
            if not self.pf['approx_lya']:
                Ja = np.sum(Ja_src, axis=0)
            
            # Molecule destruction
            #kdiss = np.sum(kdiss_src, axis=0)
                                    
            # Each is grid x absorbers, or grid x [absorbers, absorbers] for gamma
            # For Ja, just has len(grid)
            self.kwargs.update({'Gamma': Gamma, 'Heat': Heat, 'gamma': gamma})
            
            if not self.pf['approx_lya']:
                self.kwargs.update({'Ja': Ja})
                
        # Compute source independent rate coefficients
        if (not self.grid.isothermal) or (t == 0):
            self.kwargs.update(self.chem.chemnet.SourceIndependentCoefficients(data['Tk']))

        # SOLVE
        newdata = self.chem.Evolve(data, t, dt, **self.kwargs)
        
        ### 
        ## Tidy up a bit
        ###
        
        # If multiple processors at work, communicate data and timestep                                                                                          
        #if (size > 1) and (self.pf['ParallelizationMethod'] == 1):
        #    for key in newdata.keys(): 
        #        newdata[key] = MPI.COMM_WORLD.allreduce(newdata[key], newdata[key])
        #        
        #    dtphot = MPI.COMM_WORLD.allreduce(dtphot, dtphot) 
                                
        # Load balance grid for next timestep                     
        #if size > 1 and self.pf['ParallelizationMethod'] == 1: 
        #    lb = self.control.LoadBalance(dtphot)   
        #else: 
        #    lb = None      
                                                                                                                                                     
        return newdata
        
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

        