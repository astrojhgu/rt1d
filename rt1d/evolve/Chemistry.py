"""

Chemistry.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Sep 21 13:03:44 2012

Description: 

Notes: If we want to parallelize over the grid, we'll need to use different
ODE integration routines, as scipy.integrate.ode is not re-entrant :(
Maybe not - MPI should be OK, multiprocessing should cause the problems.

"""

import copy
import numpy as np
from scipy.integrate import ode
from ..util import convert_ion_name
from ..physics.Constants import k_B

#try:
#    from mpi4py import MPI
#    rank = MPI.COMM_WORLD.rank
#    size = MPI.COMM_WORLD.size
#except ImportError:
#    rank = 0
#    size = 1
    
tiny_ion = 1e-12 

class Chemistry(object):
    """ Class for evolving chemical reaction equations. """
    def __init__(self, grid, rt=False, dengo=False, atol=1e-8, rtol=1e-8,
        rate_src='fk94'):
        """
        Create a chemistry object.
        
        Parameters
        ----------
        grid: rt1d.static.Grid.Grid instance
        dengo: bool
            Use dengo?
        rt: bool
            Use radiative transfer?
            
        """

        self.grid = grid
        self.dengo = dengo
        self.rtON = rt
        
        if dengo:
            from ..static.ChemicalNetwork import \
                DengoChemicalNetwork as ChemicalNetwork
        else:
            from ..static.ChemicalNetwork import \
                SimpleChemicalNetwork as ChemicalNetwork
        
        self.chemnet = ChemicalNetwork(grid, rate_src=rate_src)
        
        self.solver = ode(self.chemnet.RateEquations, 
            jac=self.chemnet.Jacobian).set_integrator('vode',
            method='bdf', nsteps=1e4, with_jacobian=True,
            atol=atol, rtol=rtol)
            
        self.zeros_gridxq = np.zeros([self.grid.dims, len(self.grid.evolving_fields)])
        self.zeros_grid_x_abs = np.zeros_like(self.grid.zeros_grid_x_absorbers)
        self.zeros_grid_x_abs2 = np.zeros_like(self.grid.zeros_grid_x_absorbers2)
    
    #@property        
    #def load_balance(self):
    #    """
    #    Return array the same size as the grid, each element noting
    #    the processor responsible for solving chemistry in that cell. 
    #    """        
    #    
    #    if not hasattr(self, '_proc_dist'):
    #        ascending = np.arange(size)
    #        descending = list(ascending.copy())
    #        descending.reverse()
    #        single_permutation = np.concatenate((ascending, descending))
    #        
    #        D = size * 2
    #        self._proc_dist = np.zeros(self.grid.dims)
    #        for i in xrange(self.grid.dims):
    #            self._proc_dist[i] = single_permutation[i % D]
    #        
    #    return self._proc_dist
        
    def Evolve(self, data, t, dt, **kwargs):
        """
        Evolve all cells by dt.
        """
        
        if self.grid.expansion:
            z = self.grid.cosm.TimeToRedshiftConverter(0, t, self.grid.zi)
            dz = dt / self.grid.cosm.dtdz(z)
        else:
            z = dz = 0
                
        if self.dengo:
            return self.EvolveDengo(data, t, dt)
        else:
            if 'he_1' in self.grid.absorbers:
                i = self.grid.absorbers.index('he_1')
                self.chemnet.psi[...,i] *= data['he_2'] / data['he_1']
                
        # Make sure we've got number densities
        if 'n' not in data.keys():
            data['n'] = self.grid.particle_density(data, z)        
                
        newdata = {}
        for field in data:
            newdata[field] = data[field].copy()
                    
        kwargs_by_cell = self.sort_kwargs_by_cell(kwargs)
                               
        self.q_grid = np.zeros_like(self.zeros_gridxq)
        self.dqdt_grid = np.zeros_like(self.zeros_gridxq)
                              
        # Loop over grid and solve chemistry
        for cell in xrange(self.grid.dims):

            #if rank != self.load_balance[cell]:
            #    continue

            # Construct q vector
            q = np.zeros(len(self.grid.evolving_fields))
            for i, species in enumerate(self.grid.evolving_fields):
                q[i] = data[species][cell]
                                    
            kwargs_cell = kwargs_by_cell[cell]
              
            if self.rtON:
                args = (cell, kwargs_cell['Gamma'], kwargs_cell['gamma'],
                    kwargs_cell['Heat'], data['n'][cell], t)
            else:
                args = (cell, self.grid.zeros_absorbers, 
                    self.grid.zeros_absorbers2, self.grid.zeros_absorbers, 
                    data['n'][cell], t)
                                                        
            self.solver.set_initial_value(q, 0.0).set_f_params(args).set_jac_params(args)
            self.solver.integrate(dt)
                                                
            self.q_grid[cell] = q.copy()
            self.dqdt_grid[cell] = self.chemnet.dqdt.copy()

            for i, value in enumerate(self.solver.y):
                newdata[self.grid.evolving_fields[i]][cell] = self.solver.y[i]
                                                         
        # Collect results        
        #if size > 1:
        #    collected_data = {}
        #    for key in newdata:
        #        tmp = np.zeros_like(newdata[key])
        #        nothing = MPI.COMM_WORLD.Allreduce(newdata[key], tmp)
        #        collected_data[key] = tmp
        #        del tmp    
        #        
        #    newdata = collected_data
        #    tmp_q = np.zeros_like(self.q_grid)
        #    tmp_qdot = np.zeros_like(self.q_grid)    
        #    nothing = MPI.COMM_WORLD.Allreduce(self.q_grid, tmp_q)
        #    nothing = MPI.COMM_WORLD.Allreduce(self.dqdt_grid, tmp_qdot)
        #        
        #    self.q_grid = tmp_q
        #    self.dqdt_grid = tmp_qdot
                            
        # Compute particle density
        newdata['n'] = self.grid.particle_density(newdata, z - dz)
                        
        return newdata  

    def EvolveDengo(self, data, t, dt):
        """
        Solve rate equations, which evolve ion fractions and gas energy.
        """
                
        newdata = {}
        for key in data.keys():
            newdata[key] = copy.deepcopy(data[key])
       
        # Create all kwargs
        if t == 0 or not self.grid.isothermal:
            self.kwargs = {}
            for element in self.chemnet.networks:
                network = self.chemnet.networks[element]
                network.init_single_temperature(data['Tk'])
                for reaction in network.reactions.keys():
                    prefix, suffix = reaction.split('_')
                    prefix = convert_ion_name(prefix, convention='underscore')
                    val = network.reactions[reaction].coeff_fn(network)
                    self.kwargs['%s_%s' % (prefix, suffix)] = val 
                
                #for action in network.cooling_actions.keys():
                #    prefix, suffix = reaction.split('_')
                #    prefix = convert_ion_name(prefix, convention='underscore')
                #    val = network.cooling_actions[action].coeff_fn(network)
                #    self.kwargs['%s_%s' % (prefix, suffix)] = val     
                                        
        kwargs_by_cell = self.sort_kwargs_by_cell(self.kwargs)
                
        # Loop over grid and solve chemistry            
        for cell in xrange(self.grid.dims):
                                
            # Construct q vector
            q = np.zeros(len(self.grid.qmap))
            for i, species in enumerate(self.grid.qmap):
                q[i] = data[species][cell]       
            
            args = []
            for kw in kwargs_by_cell[cell]:
                args.append((kw, kwargs_by_cell[cell][kw]))
                                                                
            self.solver.set_initial_value(q, 0.0).set_f_params(args).set_jac_params(args)
            self.solver.integrate(dt)
                                    
            for i, value in enumerate(self.solver.y):
                if self.grid.qmap[i] in self.grid.all_ions:
                    limit = tiny_ion
                else:
                    limit = value
                    
                newdata[self.grid.qmap[i]][cell] = max(value, limit)
        
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
