"""

ControlSimulation.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Jan 18 13:39:42 2012

Description: Routines used to control simulation (adaptive timestep, 
load-balancing, etc.)

"""

import copy
import numpy as np
from .Cosmology import Cosmology

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1    

class ControlSimulation:
    def __init__(self, pf):
        self.pf = pf
        self.cosm = Cosmology(pf)
        
        self.GridDimensions = int(pf.GridDimensions)
        self.grid = np.arange(self.GridDimensions)
        self.R0 = pf.LengthUnits * pf.StartRadius
        self.z0 = pf.InitialRedshift
                
        self.mask = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                
    def ComputeInitialPhotonTimestep(self, data, r):
        """
        Compute photon timestep based on maximum allowed fractional change
        in hydrogen and helium neutral fractions (Shapiro et al. 2004).
        """          
        
        # Read in some data, mostly for first cell
        n_H = data['HIDensity'][0] + data['HIIDensity'][0]
        n_He = data['HeIDensity'][0] + data['HeIIDensity'][0] + data['HeIIIDensity'][0]
        n_e = data['HIIDensity'][0] + data['HeIIDensity'][0] + 2. * data['HeIIIDensity'][0]
        nabs = np.array([data['HIDensity'][0], data['HeIDensity'][0], data['HeIIDensity'][0]])
        nion = np.array([data['HIIDensity'][0], data['HeIIDensity'][0], data['HeIIIDensity'][0]])
        T = data['Temperature'][0]
        
        ncol = 1. * np.zeros(3)
        ncell = r.dx[0] * nabs    
        nout = np.log10(ncell)
        Vsh = r.coeff.ShellVolume(self.R0, r.dx[0])    
        Lbol = r.rs.BolometricLuminosity(0)
        logxHII = np.log10(nion[0] / n_H)
                                                                    
        indices_in = r.coeff.Interpolate.GetIndices([ncol[0], ncol[1], ncol[2], logxHII, 0])

        args = [nabs, nion, n_H, n_He, n_e]
        args.extend(r.coeff.ConstructArgs([nabs, nion, n_H, n_He, n_e], 
            indices_in, Lbol, self.R0, ncol, T, r.dx[0], 0., self.z0))   
                                                                          
        nabs, nion, n_H, n_He, n_e, Gamma, gamma, Beta, alpha, k_H, \
            zeta, eta, psi, xi, omega, hubble, compton = args
                                                                                                        
        # START TIMESTEP CALCULATION
        dtHI = 1e50   
        dHIdt = 1e-50   
        if self.pf.HIRestrictedTimestep and nabs[0] > 0:
            dHIdt = nabs[0] * (Gamma[0] + Beta[0] * n_e) + \
                    np.sum(gamma[0] * nabs) - \
                    nion[0] * n_e * alpha[0]
            if self.pf.CosmologicalExpansion:        
                dHIdt -= 3. * nabs[0] * hubble    
            dtHI = self.pf.MaxHIIChange * nabs[0] / abs(dHIdt)
        
        dtHeII = 1e50
        dHeIIdt = 1e-50
        if self.pf.MultiSpecies and self.HeIIRestrictedTimestep and nabs[2] > 0:
            dHeIIdt = nabs[1] * Gamma[1]            
            dtHeII = self.pf.MaxHeIIChange * nabs[2] / dHeIIdt   
                         
        dtHeIII = 1e50
        dHeIIIdt = 1e-50
        if self.pf.MultiSpecies and self.pf.HeIIIRestrictedTimestep and nion[2] > 0:                        
            dHeIIIdt = nabs[2] * Gamma[2] 
            dtHeIII = self.pf.MaxHeIIIChange * nion[2] / dHeIIdt
                  
        dtne = 1e50     
        if self.pf.ElectronRestrictedTimestep:  
            dHIIdt = nabs[0] * Gamma[0]
            dHeIIdt = nabs[1] * Gamma[1]
            dHeIIIdt = nabs[2] * Gamma[2]              
            dedt = np.abs(dHIIdt + dHeIIdt + 2. * dHeIIIdt)
            dtne = self.pf.MaxElectronChange * n_e / dedt 
            
        dtT = 1e50
        if self.pf.TemperatureRestrictedTimestep:
            dTdt = np.abs(2. * T * hubble)
            dtT = self.pf.MaxTemperatureChange * T / dTdt    

        return min(dtHI, dtHeII, dtHeIII, dtne, dtT)
    
    def ComputePhotonTimestep(self, tau, nabs, nion, ncol, n_H, n_He, n_e, n_B, 
        Gamma, gamma, Beta, alpha, k_H, zeta, eta, psi, xi, omega, hubble, compton, 
        T, z, dt):
        """
        Compute photon timestep based on maximum allowed fractional change
        in hydrogen and helium neutral fractions (Shapiro et al. 2004).
        """          
                
        dtHI = 1e50     
        if self.pf.HIRestrictedTimestep:
            dHIIdt = nabs[0] * (Gamma[0] + Beta[0] * n_e) + \
                     np.sum(gamma[0] * nabs) - \
                     nion[0] * n_e * alpha[0]
            if self.pf.CosmologicalExpansion:             
                dHIIdt -= 3. * nabs[0] * hubble
            if tau[0] >= self.pf.OpticalDepthDefiningIFront[0]:
                dtHI = self.pf.MaxHIIChange * nabs[0] / abs(dHIIdt)
        
        dtHeII = 1e50 
        if self.pf.MultiSpecies and self.pf.HeIIRestrictedTimestep:
            dHeIIdt = nabs[1] * (Gamma[1] + Beta[1] * n_e) + \
                      np.sum(gamma[1] * nabs) + \
                      alpha[2] * n_e * nion[2] - \
                      (alpha[1] + Beta[2] + xi[1]) * nion[1] * n_e
            if tau[1] >= self.pf.OpticalDepthDefiningIFront[1]:
                dtHeII = self.pf.MaxHeIIChange * nabs[2] / abs(dHeIIdt)
                
        dtHeIII = 1e50  
        if self.pf.MultiSpecies and self.pf.HeIIIRestrictedTimestep:
            dHeIIIdt = nabs[2] * (Gamma[2] + Beta[2] * n_e) + \
                       nion[2] * n_e * alpha[2]
            if tau[2] >= self.pf.OpticalDepthDefiningIFront[2]:                         
                dtHeIII = self.pf.MaxHeIIIChange * nion[2] / abs(dHeIIIdt)
              
        dtHeI = 1e50            
        if self.pf.MultiSpecies and self.pf.HeIRestrictedTimestep and (nabs[1] / n_He) > self.pf.MinimumSpeciesFraction:
            if tau[1] >= self.pf.OpticalDepthDefiningIFront[1]:
                dHeIdt = (alpha[1] + xi[1]) * nion[1] * n_e \
                    - nabs[1] * (Gamma[1] + Beta[1] * n_e)
                dtHeI = self.pf.MaxHeIChange * nabs[1] / abs(dHeIdt)        
        
        dtne = 1e50     
        if self.pf.ElectronRestrictedTimestep:  
            dHIIdt = nabs[0] * (Gamma[0] + Beta[0] * n_e) + \
                     np.sum(gamma[0] * nabs) - \
                     nion[0] * n_e * alpha[0]  
            dHeIIdt = nabs[1] * (Gamma[1] + Beta[1] * n_e) + \
                      np.sum(gamma[1] * nabs) + \
                      alpha[2] * n_e * nion[2] - \
                      (alpha[1] + Beta[2] + xi[1]) * nion[1] * n_e 
            dHeIIIdt = nabs[2] * (Gamma[2] + Beta[2] * n_e) + \
                       nion[2] * n_e * alpha[2]                  
            dedt = np.abs(dHIIdt + dHeIIdt + 2. * dHeIIIdt)
            dtne = self.pf.MaxElectronChange * n_e / dedt 
            
        dtT = 1e50
        if self.pf.TemperatureRestrictedTimestep:
            dTdt = np.abs(np.sum(k_H * nabs) - n_e * (np.sum(zeta * nabs) + \
                np.sum(eta * nabs) + np.sum(psi * nabs) + nion[2] * omega[1]) - \
                3. * hubble * T / 2.)
            dtT = self.pf.MaxTemperatureChange * T / dTdt
                        
        return min(dtHI, dtHeII, dtHeIII, dtHeI, dtne, dtT)        
        
    def LoadBalance(self, dtphot):
        """
        Return cells that should be solved by each processor.
        """    
        
        # Estimate of number of steps for each cell
        nsubsteps = 1. / dtphot
                
        # Compute CDF for timesteps
        cdf = np.cumsum(nsubsteps) / np.sum(nsubsteps)
        intervals = np.linspace(1. / size, 1, size)
                
        lb = list(np.interp(intervals, cdf, self.grid, left = 0))
        lb[-1] = self.GridDimensions
        lb.insert(0, 0)
                
        for i, entry in enumerate(lb):
            lb[i] = int(entry)
                
        # Make sure no two elements are the same - this may not always work
        while np.any(np.diff(lb) == 0):
            for i, entry in enumerate(lb[1:-1]):
                                
                if entry == 0: 
                    lb[i + 1] = entry + 1
                if entry == lb[i]:
                    lb[i + 1] = entry + 1
                                                                                            
        return lb    
    
    def DistributeDataAcrossProcessors(self, data, lb):
        """
        Setup array marking which processors solve which cells.
        """    
        
        # If parallelizing over grid, do this so an MPI all-reduce doesn't 
        # add field values together
        if self.pf.ParallelizationMethod == 1 and size > 1:
            proc_mask = np.zeros(self.GridDimensions)
            solve_arr = np.arange(self.GridDimensions)
            condition = (solve_arr >= lb[rank]) & (solve_arr < lb[rank + 1])
            proc_mask[condition] = 1
            solve_arr = solve_arr[proc_mask == 1]
        else:
            solve_arr = np.ones(self.GridDimensions)
            
        # Set up newdata dictionary                                
        newdata = {}
        for key in data.keys(): 
            newdata[key] = copy.deepcopy(data[key])
            
            if self.pf.ParallelizationMethod == 1 and size > 1:
                newdata[key][proc_mask == 0] = 0        
            
        return solve_arr, newdata        
        
         