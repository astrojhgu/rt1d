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
from ..physics.Constants import k_B

class Chemistry:
    def __init__(self, grid, dengo = False, rt = False):
        """
        Initialize chemistry solver with InitializeGrid.Grid class instance.
        """
        self.grid = grid
        self.dengo = dengo
        self.rtON = rt
        
        if dengo:
            from ..init.InitializeChemicalNetwork import \
                DengoChemicalNetwork as ChemicalNetwork
        else:
            from ..init.InitializeChemicalNetwork import \
                SimpleChemicalNetwork as ChemicalNetwork
        
        self.chemnet = ChemicalNetwork(grid)
        
        self.solver = ode(self.chemnet.RateEquations, 
            jac = self.chemnet.Jacobian).set_integrator('vode', 
            method = 'bdf', nsteps = 1e4, with_jacobian = False,
            atol = 1e-12, rtol = 1e-8)
            
        self.zeros_gridxq = np.zeros([self.grid.dims, len(self.grid.all_species)])
        self.zeros_grid_x_abs = np.zeros_like(self.grid.zeros_grid_x_absorbers)
        self.zeros_grid_x_abs2 = np.zeros_like(self.grid.zeros_grid_x_absorbers2)
            
    def Evolve(self, data, dt, **kwargs):
        """
        Evolve all cells by dt.
        """
                
        if self.dengo:
            return self.EvolveDengo(data, dt)
        else:
            if 'he_1' in self.grid.absorbers:
                i = self.grid.absorbers.index('he_1')
                self.chemnet.psi[...,i] *= data['he_2'] / data['he_1']
                
        newdata = {}
        for species in data:
            newdata[species] = data[species].copy()
               
        kwargs_by_cell = self.sort_kwargs_by_cell(kwargs)
                               
        self.q_grid = np.zeros_like(self.zeros_gridxq)
        self.dqdt_grid = np.zeros_like(self.zeros_gridxq)
               
        # Loop over grid and solve chemistry
        for cell in xrange(self.grid.dims):

            # Construct q vector
            q = np.zeros(len(self.grid.all_species))
            for i, species in enumerate(self.grid.all_species):
                q[i] = data[species][cell]
                
            kwargs_cell = kwargs_by_cell[cell]
            
            if self.rtON:
                args = (cell, kwargs_cell['Gamma'], kwargs_cell['gamma'],
                    kwargs_cell['Heat'], data['n'][cell])
            else:
                args = (cell, self.grid.zeros_absorbers, self.grid.zeros_absorbers2, 
                    self.grid.zeros_absorbers)
                            
            self.solver.set_initial_value(q, 0.0).set_f_params(args).set_jac_params(args)
            self.solver.integrate(dt)
            
            self.q_grid[cell] = q.copy()
            self.dqdt_grid[cell] = self.chemnet.dqdt.copy()

            for i, value in enumerate(self.solver.y):
                newdata[self.grid.all_species[i]][cell] = self.solver.y[i]
                
        # Convert ge to T
        if not self.grid.isothermal:
            newdata['n'] = self.grid.particle_density(newdata)
            newdata['T'] = newdata['ge'] * 2. / 3. / k_B / newdata['n']
        
        return newdata

    def EvolveDengo(self, data, dt):
        """
        Solve rate equations, which evolve ion fractions and gas energy.
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
                                        
        kwargs_by_cell = self.sort_kwargs_by_cell(kwargs)
        
        # Loop over grid and solve chemistry            
        for cell in xrange(self.grid.dims):
                                
            # Construct q vector
            q = np.zeros(len(self.grid.qmap))
            for i, species in enumerate(self.grid.qmap):
                q[i] = data[species][cell]            
                        
            self.solver.set_initial_value(q, 0.0).set_f_params(kwargs_by_cell[cell]).set_jac_params(kwargs_by_cell[cell])
            self.solver.integrate(dt)
            new_q = self.solver.y 
                                    
            for i, value in enumerate(new_q):
                newdata[self.grid.qmap[i]][cell] = value
        
        return newdata
    
    def sort_kwargs_by_cell(self, kwargs):
        """
        Convert kwargs dictionary to list - entries correspond to cells, a dictionary
        of values for each.
        """    
        
        # Organize by cell                        
        kwargs_by_cell = []
        for cell in xrange(self.grid.dims):
            new_kwargs = {}
            for key in kwargs.keys():
                new_kwargs[key] = kwargs[key][cell]
                
            kwargs_by_cell.append(new_kwargs)
        
        return kwargs_by_cell
