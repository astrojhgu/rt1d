"""

ComputeTimestep.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Dec 27 14:12:50 2012

Description: 

"""

import numpy as np
from ..util.Warnings import dt_error

huge_dt = 1e30  # seconds

class ComputeTimestep:
    def __init__(self, grid, epsilon=0.1):
        self.grid = grid
        self.epsilon = epsilon
        self.grid_indices = np.arange(self.grid.dims)
    
    def Limit(self, q, dqdt, z=None, tau=None, tau_ifront=0.5, 
        method=['ions']):
        """
        Limit timestep based on maximum allowed change in fields.  Which 
        fields determined by method parameter.
        """
        
        # Projected timestep for each cell and field (dt.shape = grid x species)
        dt = self.epsilon * q / np.abs(dqdt)
        
        # Don't let dt -> 0 where species fraction is zero
        dt[np.logical_and(q == 0, self.grid.types >= 0)] = huge_dt
        
        # What's limiting the time-step?
                
        # Isolate cells beyond I-front
        if tau is not None:
            dt[tau <= tau_ifront, ...] = huge_dt
                
        tries = []  
        new_dt = huge_dt
        for mth in method:
        
            # Determine index correspond to element(s) of q to use to limit dt
            if mth == 'ions':
                j = self.grid.types == 1
            elif mth == 'neutrals':
                j = self.grid.types == 0
            elif mth == 'electrons':
                j = self.grid.evolving_fields.index('de')
            elif mth == 'temperature':
                if 'Tk' in self.grid.evolving_fields:
                    j = self.grid.evolving_fields.index('Tk')
                else:
                    min_dt = huge_dt
            elif mth == 'hubble' and self.grid.expansion:
                min_dt = self.epsilon * self.grid.cosm.HubbleTime(z)
            else:
                raise ValueError('Unrecognized dt restriction method: %s' % mth)
            
            min_dt = np.min(dt[..., j])
            
            # Determine which cell is behaving badly
            if self.grid.dims == 1:
                tries.append(0)
            else:
                if j is not None:
                    which_cell = int(self.grid_indices[np.argwhere(dt[...,j] == min_dt)].squeeze())
                else:
                    which_cell = 0
                
                tries.append(which_cell)

            # Update the time-step
            new_dt = min(new_dt, min_dt)

        # Raise error if any funny stuff happens                        
        if (new_dt <= 0) or np.isnan(new_dt) or np.isinf(new_dt):
            dt_error(self.grid, q, dqdt, new_dt, tries, method)

        return new_dt    

        
    
     
