"""

ComputeTimestep.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Dec 27 14:12:50 2012

Description: 

"""

import numpy as np

huge_dt = 1e30  # seconds

class ComputeTimestep:
    def __init__(self, grid, epsilon = 0.1):
        self.grid = grid
        self.epsilon = epsilon
    
    def Limit(self, q, dqdt, z=None, tau=None, tau_ifront=0.5, 
        method = ['ions']):
        """
        Limit timestep based on maximum allowed change in fields.  Which 
        fields determined by method parameter.
        """
        
        # Projected timestep for each cell and field (dt.shape = grid x species)
        dt = self.epsilon * q / np.abs(dqdt)
        
        # Isolate cells beyond I-front
        if tau is not None:
            dt[tau <= tau_ifront, ...] = huge_dt
                
        new_dt = huge_dt
        for mth in method:
        
            if mth == 'ions':
                new_dt = min(new_dt, 
                    np.min(dt[..., self.grid.types == 1]))
            elif mth == 'neutrals':
                new_dt = min(new_dt, 
                    np.min(dt[..., self.grid.types == 0]))
            elif mth == 'electrons':
                new_dt = min(new_dt, 
                    np.min(dt[..., self.grid.all_species.index('de')]))
            elif mth == 'temperature' and 'T' in self.grid.all_species:
                new_dt = min(new_dt, 
                    np.min(dt[..., self.grid.all_species.index('T')]))
            elif mth == 'hubble' and self.grid.expansion:
                new_dt = min(new_dt, self.epsilon \
                    * self.grid.cosm.HubbleTime(z))
            else:
                new_dt = min(new_dt, np.min(dt))
                
        # Raise error if any funny stuff happens
        if new_dt < 0: 
            raise ValueError('ERROR: dt < 0.') 
        elif new_dt == 0:
            raise ValueError('ERROR: dt = 0.')  
        elif np.isnan(new_dt):  
            raise ValueError('ERROR: dt -> inf.')      
        
        return new_dt    
            
        
    
     
