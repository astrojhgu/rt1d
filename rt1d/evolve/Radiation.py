"""

Radiation.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Sep 21 13:03:52 2012

Description: Compute properties of the radiation field, pass to Radiate, 
which calls chemistry solver.

"""

import numpy as np
from ..util import parse_kwargs
from .Chemistry import Chemistry
from ..physics.Constants import *
from .RadiationField import RadiationField

class Radiation:
    def __init__(self, grid, sources, **kwargs):
        self.pf = parse_kwargs(**kwargs)
        self.grid = grid
        self.srcs = sources
                
        # Initialize chemistry network / solver
        self.chem = Chemistry(grid, rt = kwargs['radiative_transfer'])
        
        # Initialize RT solver
        if self.srcs is not None:
            self.rfield = RadiationField(grid, sources, **kwargs)
        
    @property
    def finite_c(self):
        if self.pf['infinite_c']:
            return False
        
        return True
    
    def Evolve(self, data, t, dt, z=None, **kwargs):
        """
        This routine calls our solvers and updates 'data' -> 'newdata'
        """
        
        # Make data globally accessible
        self.data = data
                
        # Figure out which processors will solve which cells and create newdata dict
        #self.solve_arr, newdata = self.control.DistributeDataAcrossProcessors(data, lb)
        
        # Compute source dependent rate coefficients
        self.kwargs = {}
        if self.pf['radiative_transfer']:
            Gamma_src, gamma_src, Heat_src = \
                self.rfield.SourceDependentCoefficients(data, t, z, **kwargs)

            if len(self.srcs) > 1:
                for i, src in enumerate(self.srcs):
                    self.kwargs.update({'Gamma_%i' % i: Gamma_src[i], 
                        'gamma_%i' % i: gamma_src[i],
                        'Heat_%i' % i: Heat_src[i]})
            
            Gamma = np.sum(Gamma_src, axis=0)
            gamma = np.sum(gamma_src, axis=0)
            Heat = np.sum(Heat_src, axis=0)
                        
            # Each is grid x absorbers, or grid x [absorbers, absorbers] for gamma
            self.kwargs.update({'Gamma': Gamma, 'Heat': Heat, 'gamma': gamma})
                
        # Compute source independent rate coefficients
        if (not self.grid.isothermal) or (t == 0):
            self.kwargs.update(self.chem.chemnet.SourceIndependentCoefficients(data['Tk']))

        # SOLVE
        newdata = self.chem.Evolve(data, t, dt, **self.kwargs)
                
        ### 
        ## Tidy up a bit
        ###
        
        # If multiple processors at work, communicate data and timestep                                                                                          
        #if (size > 1) and (self.pf['ParallelizationMethod'] == 1):
        #    for key in newdata.keys(): 
        #        newdata[key] = MPI.COMM_WORLD.allreduce(newdata[key], newdata[key])
        #        
        #    dtphot = MPI.COMM_WORLD.allreduce(dtphot, dtphot) 
                                
        # Load balance grid for next timestep                     
        #if size > 1 and self.pf['ParallelizationMethod'] == 1: 
        #    lb = self.control.LoadBalance(dtphot)   
        #else: 
        #    lb = None      
                                                                                                                                                     
        return newdata

        