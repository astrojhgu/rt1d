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
    def __init__(self, grid, source, **kwargs):
        self.pf = parse_kwargs(**kwargs)
        self.grid = grid
        self.src = source
        
        # Initialize chemistry network / solver
        self.chem = Chemistry(grid)
        
        # Initialize RT solver
        self.rfield = RadiationField(grid, source, **kwargs)
        
        # Pick RT solver
        #if self.finite_c:
        #    self.Evolve = self.rad.EvolvePhotonsAtFiniteSpeed
        #else:
        #    self.Evolve = self.rad.EvolvePhotonsAtInfiniteSpeed   
        
    @property
    def finite_c(self):
        """
        Speed of light.
        """
        
        if self.pf['InfiniteSpeedOfLight']:
            return False
        
        return True
    
    def Evolve(self, data, t, dt):
        """
        This routine calls our solvers and updates 'data' -> 'newdata'
        """
        
        # Make data globally accessible
        self.data = data
        
        # Figure out which processors will solve which cells and create newdata dict
        #self.solve_arr, newdata = self.control.DistributeDataAcrossProcessors(data, lb)
                          
        # If we're in an expanding universe, prepare to dilute densities by (1 + z)**3
        #self.z = 0 
        #self.dz = 0   
        #if self.pf['CosmologicalExpansion']: 
        #    self.z = self.cosm.TimeToRedshiftConverter(0., t, self.pf['InitialRedshift'])
        #    self.dz = dt / self.cosm.dtdz(self.z)
               
        
                                                                                                     
        # Retrieve indices used for N-D interpolation
        #self.indices_all = []
        #for i, col in enumerate(self.ncol_all):
        #    tmp = []
        #    for rs in self.rs.all_sources:
        #        if rs.TableAvailable:
        #            tmp.append(rs.Interpolate.GetIndices([col[0], col[1], col[2], np.log10(self.x_HII_arr[i]), t]))
        #        else:
        #            tmp.append(None)
        #    self.indices_all.append(tmp)
        #                                        
        ## Compute tau *between* source and all cells
        #tau_all_arr = self.ComputeOpticalDepths([self.ncol_HI, self.ncol_HeI, self.ncol_HeII])
        #self.tau_all = zip(*tau_all_arr)
        
        # Compute source dependent rate coefficients
        Gamma, gamma, Heat = self.rfield.SourceDependentCoefficients(data, t)            

        kwargs = {'Gamma': Gamma, 'Heat': Heat, 'gamma': gamma}
                
        # Compute source independent rate coefficients
        if (not self.grid.isothermal) or (t == 0):
            kwargs.update(self.chem.chemnet.SourceIndependentCoefficients(data['T']))

        # SOLVE
        newdata = self.chem.Evolve(data, dt, **kwargs)
                
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

        