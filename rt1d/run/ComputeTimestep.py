"""

ComputeTimestep.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Dec 27 14:12:50 2012

Description: 

"""

import numpy as np

class ComputeTimestep:
    def __init__(self, grid, pf = None):
        self.pf = pf
        self.grid = grid
        
    def IonLimited(self, q, dqdt, epsilon = 0.1, limit = 'ions'):
        if limit == 'ions':
            return self.IonizedFractionLimited(q, dqdt, epsilon = 0.1)
        else:
            return self.NeutralFractionLimited(q, dqdt, epsilon = 0.1)
        
    def EnergyLimited(self, q, dqdt, epsilon = 0.1):
        """
        Compute next timestep based on maximum allowed change ('epsilon')
        in neutral fractions.
        """
        dt = epsilon * q / np.abs(dqdt)
        return dt[self.grid.all_species.index('ge')]           
    
    def ElectronLimited(self, q, dqdt, epsilon = 0.1):
        """
        Compute next timestep based on maximum allowed change ('epsilon')
        in neutral fractions.
        """
        dt = epsilon * q / np.abs(dqdt)
        return dt[self.grid.all_species.index('de')]               
        
    def IonizedFractionLimited(self, q, dqdt, epsilon = 0.1):
        """
        Compute next timestep based on maximum allowed change ('epsilon')
        in ionized fractions.
        """        
        dt = epsilon * q / np.abs(dqdt)
        return np.max(dt[self.grid.types == 1])
    
    def NeutralFractionLimited(self, q, dqdt, epsilon = 0.1):
        """
        Compute next timestep based on maximum allowed change ('epsilon')
        in neutral fractions.
        """
        dt = epsilon * q / np.abs(dqdt)
        return np.max(dt[self.grid.types == 0])
    
    def EvolutionLimited(self, q, dqdt, epsilon = 0.1):
        """
        Compute next timestep based on maximum allowed change ('epsilon')
        in all evolving species.
        """
        dt = epsilon * q / np.abs(dqdt)
        return np.max(dt)
    
     
