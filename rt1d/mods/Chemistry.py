"""

Chemistry.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Sep 21 13:03:44 2012

Description: 

"""

import copy
import pylab as pl
import numpy as np
from scipy.integrate import ode
from ..init.InitializeChemicalNetwork import ChemicalNetworkThing

class Chemistry:
    def __init__(self, grid):
        """
        Initialize chemistry solver with InitializeGrid.Grid class instance.
        """
        self.grid = grid
        self.chemnet = ChemicalNetworkThing(grid)
        
        self.solver = ode(self.chemnet.RateEquations, 
            jac = self.chemnet.Jacobian).set_integrator('vode', 
            method = 'bdf', nsteps = 1e4, with_jacobian = True,
            atol = 1e-12, rtol = 1e-8)
                    
    def Evolve(self, data, dt):
        """
        Solve rate equations which evolve ion fractions and gas energy.
        """    
        
        newdata = {}
        for key in data.keys(): 
            newdata[key] = copy.deepcopy(data[key])
       
        # Create all kwargs
        kwargs = {} 
        for element in self.chemnet.networks:
            network = self.chemnet.networks[element]
            network.init_single_temperature(data['T'])
            for reaction in network.reactions.keys():
                val = network.reactions[reaction].coeff_fn(network)
                kwargs[reaction] = val
                        
        # Organize by cell                        
        kwargs_list = []
        for cell in xrange(self.grid.dims):
            new_kwargs = {}
            for key in kwargs.keys():
                new_kwargs[key] = kwargs[key][cell]
                
            kwargs_list.append(new_kwargs)
            
        # Loop over grid and solve chemistry            
        for cell in xrange(self.grid.dims):
                                
            # Construct q vector
            q = np.zeros(len(self.grid.qmap))
            for i, species in enumerate(self.grid.qmap):
                q[i] = data[species][cell]
            
            self.solver.set_initial_value(q, 0.0).set_f_params(kwargs_list[cell]).set_jac_params(kwargs_list[cell])
            self.solver.integrate(dt)
            new_q = self.solver.y 
                                    
            for i, value in enumerate(new_q):
                newdata[self.grid.qmap[i]][cell] = value
        
        return newdata


