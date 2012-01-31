"""

ControlSimulation.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Jan 18 13:39:42 2012

Description: Routines used to control simulation (adaptive timestep, 
load-balancing, etc.)

"""

import numpy as np

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
        
        self.GridDimensions = self.pf["GridDimensions"]
        self.grid = np.arange(self.GridDimensions)
        
        self.MultiSpecies = pf["MultiSpecies"]
        self.MaxHIIChange = pf["MaxHIIChange"]
        self.MaxHeIIChange = pf["MaxHeIIChange"]
        self.MaxHeIIIChange = pf["MaxHeIIIChange"]
        self.MaxElectronChange = pf["MaxElectronChange"]
        self.HIRestrictedTimestep = pf["HIRestrictedTimestep"]
        self.HeIRestrictedTimestep = pf["HeIRestrictedTimestep"]
        self.HeIIRestrictedTimestep = pf["HeIIRestrictedTimestep"]
        self.OpticalDepthOfIFront = pf["OpticalDepthOfIFront"]
        self.ElectronFractionRestrictedTimestep = pf["ElectronFractionRestrictedTimestep"]
                
    def ComputeInitialPhotonTimestep(self, tau, Gamma, gamma, Beta, alpha, nabs, nion, ncol, n_H, n_He, n_e, force):
        """
        Compute photon timestep based on maximum allowed fractional change
        in hydrogen and helium neutral fractions (Shapiro et al. 2004).
        """          
                
        dtHI = 1e50        
        if tau[0] >= 0.5 or force[0]:
                                    
            dtHI = self.MaxHIIChange * nabs[0] / \
                np.abs(nabs[0] * Gamma[0] - nion[0] * n_e * alpha[0])
        
        dtHeI = 1e50
        if self.MultiSpecies and self.HeIRestrictedTimestep:
            
            if tau[1] >= 0.5 or force[1]:
                xHeII = nion[1] / n_He
                
                # Analogous to Shapiro et al. 2004 but for HeII
                dtHeI = self.MaxHeIIChange * nabs[1] / \
                    np.abs(nabs[1] * Gamma[1] - nion[1] * n_e * alpha[1]) 
                
        dtHeII = 1e50
        if self.MultiSpecies and self.HeIIRestrictedTimestep:
                        
            if tau[2] >= 0.5 or force[2]:
                xHeIII = nion[2] / n_He
                         
                # Analogous to Shapiro et al. 2004 but for HeIII
                dtHeII = self.MaxHeIIIChange * nabs[2] / \
                    np.abs(nabs[2] * Gamma[2] - nion[2] * n_e * alpha[2]) 
                  
        return min(dtHI, dtHeI, dtHeII)
        
    def ComputePhotonTimestep(self, tau, nabs, nion, n_H, n_He, n_e, n_B, qnew, dt):
        """
        Compute photon timestep based on maximum allowed fractional change
        in hydrogen and helium neutral fractions (Shapiro et al. 2004).
        """          
        
        dtHI = 1e50        
        if self.HIRestrictedTimestep:
            dHIdt = np.abs(((n_H - qnew[0]) - nabs[0])) / dt
            if tau[0] >= self.OpticalDepthOfIFront[0]:
                dtHI = self.MaxHIIChange * nabs[0] / dHIdt
        
        dtHeI = 1e50
        if self.MultiSpecies and self.HeIRestrictedTimestep:
            dHeIdt = np.abs(((n_He - qnew[1] - qnew[2]) - nabs[1])) / dt            
            if tau[1] >= self.OpticalDepthOfIFront[1]:
                dtHeI = self.MaxHeIIChange * nabs[1] / dHeIdt
                
        dtHeII = 1e50
        if self.MultiSpecies and self.HeIRestrictedTimestep:
            dHeIIdt = np.abs((qnew[1] - nabs[2]) / dt)        
            if tau[2] >= self.OpticalDepthOfIFront[2]:                         
                dtHeII = self.MaxHeIIIChange * nabs[2] / dHeIIdt
            
        # Change in electron fraction (relative to all baryons)    
        dtef = 1e50  
        if self.ElectronFractionRestrictedTimestep:
            n_e_new = ((qnew[0]) + qnew[1] + 2.0 * qnew[2])
            n_B_new = n_e_new + n_H + n_He
            defdt = np.abs((n_e_new / n_B_new)  - (n_e / n_B)) / dt
            dtef = self.MaxElectronChange * (n_e / n_B) / defdt 
                                                
        return min(dtHI, dtHeI, dtHeII, dtef)    
        
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