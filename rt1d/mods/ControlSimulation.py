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
        
        self.GridDimensions = pf["GridDimensions"]
        self.grid = np.arange(self.GridDimensions)
        self.LengthUnits = pf["LengthUnits"]
        self.StartRadius = pf["StartRadius"]
        self.R0 = self.LengthUnits * self.StartRadius
        
        self.MultiSpecies = pf["MultiSpecies"]
        self.MaxHIIChange = pf["MaxHIIChange"]
        self.MaxHeIChange = pf["MaxHeIChange"]
        self.MaxHeIIChange = pf["MaxHeIIChange"]
        self.MaxHeIIIChange = pf["MaxHeIIIChange"]
        self.MaxElectronChange = pf["MaxElectronChange"]
        self.HIRestrictedTimestep = pf["HIRestrictedTimestep"]
        self.HeIRestrictedTimestep = pf["HeIRestrictedTimestep"]
        self.HeIIRestrictedTimestep = pf["HeIIRestrictedTimestep"]
        self.HeIIIRestrictedTimestep = pf["HeIIIRestrictedTimestep"]
        self.OpticalDepthDefiningIFront = pf["OpticalDepthDefiningIFront"]
        self.ElectronFractionRestrictedTimestep = pf["ElectronFractionRestrictedTimestep"]
        self.MinimumSpeciesFraction = pf["MinimumSpeciesFraction"]
        
        self.mask = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                
    def ComputeInitialPhotonTimestep(self, data, r, itabs = None):
        """
        Compute photon timestep based on maximum allowed fractional change
        in hydrogen and helium neutral fractions (Shapiro et al. 2004).
        """          
        
        # Read in some data, mostly for first cell
        n_H = data['HIDensity'][0] + data['HIIDensity'][0]
        n_He = data['HeIDensity'][0] + data['HeIIDensity'][0] + data['HeIIIDensity'][0]
        n_e = data['HIIDensity'][0] + data['HeIIDensity'][0] + 2. * data['HeIIIDensity'][0]
        
        tau1 = ncol = 1. * np.ones(3)
        tau0 = gamma = Beta = alpha = nion = xi = 1. * np.zeros(3)
        nabs = np.array([data['HIDensity'][0], data['HeIDensity'][0], data['HeIIDensity'][0]])
        ncell = r.dx[0] * nabs    
        nout = ncell             
        Vsh = r.coeff.ShellVolume(self.R0, r.dx[0])    
        
        if self.pf["PhotonConserving"]:                  
            A = r.rs.BolometricLuminosity(0) / nabs / Vsh
        else:
            A = 3 * [r.rs.BolometricLuminosity(0) / 4. / np.pi / self.R0**2]
                                                                            
        indices_in = None
        indices_out = None
        if self.pf['MultiSpecies'] > 0 and self.pf['TabulateIntegrals']: 
            indices_in = r.coeff.Interpolate.GetIndices3D(ncol)  
            indices_out = r.coeff.Interpolate.GetIndices3D(nout)
            
        Gamma = np.zeros(3)                                    
        if itabs is None:
            for i, E in enumerate(r.rs.E):
                Gamma[0] += r.coeff.PhotoIonizationRate(E = E, Qdot = r.rs.IonizingPhotonLuminosity(i = i),
                    ncol = ncol, nout = nout * self.mask[0], nabs = nabs, r = self.R0,
                    dr = r.dx[0], species = 0, tau = tau0, Lbol = r.rs.BolometricLuminosity(0))[0]
                
                if self.pf['MultiSpecies']:
                    Gamma[1] += r.coeff.PhotoIonizationRate(E = E, Qdot = r.rs.IonizingPhotonLuminosity(i = i),
                        ncol = ncol, nout = nout * self.mask[0], nabs = nabs, r = self.R0,
                        dr = r.dx[0], species = 1, tau = tau0, Lbol = r.rs.BolometricLuminosity(0))[0]
                    Gamma[2] += r.coeff.PhotoIonizationRate(E = E, Qdot = r.rs.IonizingPhotonLuminosity(i = i),
                        ncol = ncol, nout = nout * self.mask[0], nabs = nabs, r = self.R0,
                        dr = r.dx[0], species = 2, tau = tau0, Lbol = r.rs.BolometricLuminosity(0))[0]
                    
        else:                     
            Gamma[0] = r.coeff.PhotoIonizationRate(species = 0, Lbol = r.rs.BolometricLuminosity(0),
                indices_in = indices_in, indices_out = indices_out, r = self.R0, dr = r.dx[0],
                nabs = nabs, ncol = ncol, tau = tau0, nout = nout * self.mask[0], A = A[0])[0]
                
            if self.pf['MultiSpecies']:
                Gamma[1] = r.coeff.PhotoIonizationRate(species = 1, Lbol = r.rs.BolometricLuminosity(0),
                    indices_in = indices_in, indices_out = indices_out, r = self.R0, dr = r.dx[0],
                    nabs = nabs, ncol = ncol, tau = tau0, nout = nout * self.mask[1], A = A[1])[0]
                Gamma[2] = r.coeff.PhotoIonizationRate(species = 2, Lbol = r.rs.BolometricLuminosity(0),
                    indices_in = indices_in, indices_out = indices_out, r = self.R0, dr = r.dx[0],
                    nabs = nabs, ncol = ncol, tau = tau0, nout = nout * self.mask[2], A = A[2])[0] 
        
        # START TIMESTEP CALCULATION
        dtHI = 1e50   
        dHIdt = 1e-50   
        if self.HIRestrictedTimestep and nabs[0] > 0:
            dHIdt = nabs[0] * Gamma[0]
            dtHI = self.MaxHIIChange * nabs[0] / dHIdt
        
        dtHeII = 1e50
        dHeIIdt = 1e-50
        if self.MultiSpecies and self.HeIIRestrictedTimestep and nabs[2] > 0:
            dHeIIdt = nabs[1] * Gamma[1]            
            dtHeII = self.MaxHeIIChange * nabs[2] / dHeIIdt   
                         
        dtHeIII = 1e50
        dHeIIIdt = 1e-50
        if self.MultiSpecies and self.HeIIIRestrictedTimestep and nion[2] > 0:                        
            dHeIIIdt = nabs[2] * Gamma[2] 
            dtHeIII = self.MaxHeIIIChange * nion[2] / dHeIIdt
                  
        dtef = 1e50  
        if self.ElectronFractionRestrictedTimestep:
            defdt = np.abs(dHIdt + dHeIIdt + 2. * dHeIIIdt)
            dtef = self.MaxElectronChange * n_e / n_B / defdt 
        
        return min(dtHI, dtHeII, dtHeIII, dtef)
    
    def ComputePhotonTimestep(self, tau, nabs, nion, ncol, n_H, n_He, n_e, n_B, Gamma, gamma, Beta, alpha, xi):
        """
        Compute photon timestep based on maximum allowed fractional change
        in hydrogen and helium neutral fractions (Shapiro et al. 2004).
        """          
                
        dtHI = 1e50        
        if self.HIRestrictedTimestep:
            dHIIdt = nabs[0] * (Gamma[0] + gamma[0] + Beta[0] * n_e) \
                - nion[0] * n_e * alpha[0]
            if tau[0] >= self.OpticalDepthDefiningIFront[0]:
                dtHI = self.MaxHIIChange * nabs[0] / abs(dHIIdt)
        
        dtHeII = 1e50
        if self.MultiSpecies and self.HeIIRestrictedTimestep:
            dHeIIdt = nabs[1] * (Gamma[1] + gamma[1] + Beta[1] * n_e) \
                + alpha[2] * n_e * nion[2] \
                - (alpha[1] + Beta[2]) * nion[1] * n_e \
                - xi[1] * n_e * nion[2]
            if tau[1] >= self.OpticalDepthDefiningIFront[1]:
                dtHeII = self.MaxHeIIChange * nabs[2] / abs(dHeIIdt)
                
        dtHeIII = 1e50
        if self.MultiSpecies and self.HeIIIRestrictedTimestep:
            dHeIIIdt = nabs[2] * (Gamma[2] + gamma[2] + Beta[2] * n_e) \
                - nion[2] * n_e * alpha[2]
            if tau[1] >= self.OpticalDepthDefiningIFront[1]:                         
                dtHeIII = self.MaxHeIIIChange * nion[2] / abs(dHeIIIdt)
              
        dtHeI = 1e50        
        if self.MultiSpecies and self.HeIRestrictedTimestep and (nabs[1] / n_He) > self.MinimumSpeciesFraction:
            if tau[1] <= self.OpticalDepthDefiningIFront[1]:
                dHeIdt = -(dHeIIdt + dHeIIIdt)
                dtHeI = self.MaxHeIChange * nabs[1] / abs(dHeIdt)        
            
        dtef = 1e50  
        if self.ElectronFractionRestrictedTimestep:
            defdt = np.abs(dHIIdt + dHeIIdt + 2. * dHeIIIdt)
            dtef = self.MaxElectronChange * (n_e / n_B) / defdt 
        
        return min(dtHI, dtHeI, dtHeII, dtHeIII, dtef)        
        
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
        
    def ComputePhotonTimestepOLD(self, tau, nabs, nion, n_H, n_He, n_e, n_B, qnew, dt):
        """
        Compute photon timestep based on maximum allowed fractional change
        in hydrogen and helium neutral fractions (Shapiro et al. 2004).
        """          
        
        dtHI = 1e50        
        if self.HIRestrictedTimestep:
            dHIdt = np.abs(((n_H - qnew[0]) - nabs[0])) / dt
            if tau[0] >= self.OpticalDepthDefiningIFront[0]:
                dtHI = self.MaxHIIChange * nabs[0] / dHIdt
        
        dtHeI = 1e50
        if self.MultiSpecies and self.HeIRestrictedTimestep:
            dHeIdt = np.abs(((n_He - qnew[1] - qnew[2]) - nabs[1])) / dt            
            if tau[1] >= self.OpticalDepthDefiningIFront[1]:
                dtHeI = self.MaxHeIIChange * nabs[1] / dHeIdt
                
        dtHeII = 1e50
        if self.MultiSpecies and self.HeIRestrictedTimestep:
            dHeIIdt = np.abs((qnew[1] - nabs[2]) / dt)        
            if tau[2] >= self.OpticalDepthDefiningIFront[2]:                         
                dtHeII = self.MaxHeIIIChange * nabs[2] / dHeIIdt
            
        # Change in electron fraction (relative to all baryons)    
        dtef = 1e50  
        if self.ElectronFractionRestrictedTimestep:
            n_e_new = ((qnew[0]) + qnew[1] + 2.0 * qnew[2])
            n_B_new = n_e_new + n_H + n_He
            defdt = np.abs((n_e_new / n_B_new)  - (n_e / n_B)) / dt
            dtef = self.MaxElectronChange * (n_e / n_B) / defdt 
        
        return min(dtHI, dtHeI, dtHeII, dtef)        