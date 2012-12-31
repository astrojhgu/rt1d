"""

Radiation.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Sep 21 13:03:52 2012

Description: 

"""

from .Radiate import *
from .Chemistry import Chemistry
from ..physics.Constants import *

class Radiation:
    def __init__(self, grid, source, **kwargs):
        self.grid = grid
        self.src = source
        
        # Initialize chemistry network / solver
        self.chem = rt1d.Chemistry(grid)
        
    @property
    def light(self):
        """
        Speed of light.
        """
        pass
    
    def Evolve(self, data, dt):
        """
        Evolve the radiation (and chemistry).
        """
        pass
    
        