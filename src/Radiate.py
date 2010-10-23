"""
Radiate.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-08-18.

Description: This routine essentially runs the show.  The method 'EvolvePhotons' is the
driver of rt1d, calling our solvers which call all the various physics modules.
     
"""

class Radiate:
    def __init__(self, pf):
        pass
        
    def EvolvePhotons(self, data, t, dt):
        """
        This routine calls our solvers, updates the values in 'data', and 
        computes what the next timestep ought to be.  Easier said than done.
        """
        return data, 0.01
        
    