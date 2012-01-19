"""

ControlSimulation.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Jan 18 13:39:42 2012

Description: Routines used to control simulation (adaptive timestep, 
load-balancing, etc.)

"""

import numpy as np

class ControlSimulation:
    def __init__(self, pf):
        self.pf = pf
        
        self.MultiSpecies = pf["MultiSpecies"]
        self.MaxHIIChange = pf["MaxHIIChange"]
        self.MaxHeIIChange = pf["MaxHeIIChange"]
        self.MaxHeIIIChange = pf["MaxHeIIIChange"]
        self.HIIRestrictedTimestep = pf["HIIRestrictedTimestep"]
        self.HeIIRestrictedTimestep = pf["HeIIRestrictedTimestep"]
        self.HeIIIRestrictedTimestep = pf["HeIIIRestrictedTimestep"]
                
    def ComputePhotonTimestep(self, tau, Gamma, gamma, Beta, alpha, nabs, nion, ncol, n_H, n_He, n_e):
        """
        Compute photon timestep based on maximum allowed fractional change
        in hydrogen and helium neutral fractions (Shapiro et al. 2004).
        """          
        
        dtHI = 1e50        
        if tau[0] >= 0.5:
                                    
            dtHI = self.MaxHIIChange * nabs[0] / \
                np.abs(nabs[0] * (Gamma[0] + gamma[0] + n_e * Beta[0]) - nion[0] * n_e * alpha[0])
        
        dtHeI = 1e50
        if self.MultiSpecies and self.HeIIRestrictedTimestep:
            
            if tau[1] >= 0.5:
                xHeII = nion[1] / n_He
                
                # Analogous to Shapiro et al. 2004 but for helium
                dtHeI = self.MaxHeIIChange * nabs[1] / \
                    np.abs(nabs[1] * (Gamma[1] + gamma[1] + n_e * Beta[1]) - nion[1] * n_e * alpha[1]) 
                
        dtHeII = 1e50
        if self.MultiSpecies and self.HeIIIRestrictedTimestep:
                        
            if tau[2] >= 0.5:
                xHeIII = nion[2] / n_He
                
                         
                alpha = 3.36e-10 * T**-0.5 * (T / 1e3)**-0.2 * (1. + (T / 4.e6)**0.7)**-1        # To n >= 1
                if T < 2.2e4: alpha *= (1.11 - 0.044 * np.log(T))                                # To n >= 2
                else: alpha *= (1.43 - 0.076 * np.log(T))
                
                # Analogous to Shapiro et al. 2004 but for helium
                dtHeI = self.MaxHeIIIChange * n_HeII / \
                    np.abs(n_HeII * (Gamma + n_e * Beta) - \
                    n_HeIII * n_e * alpha) 
        
        
        return min(dtHI, dtHeI, dtHeII)
        
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