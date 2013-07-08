"""

DiffuseSource.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Jul  8 10:34:59 MDT 2013

Description: 

"""

from ..util import evolve

class DiffuseSource(object):
    """ Class for prescribed ionizing/heating backgrounds. """
    def __init__(self, pf, src_pars, spec_pars):
        """
        Initialize diffuse radiation background.
        
        Parameters
        ----------
        pf: dict
            Full parameter file.
        src_pars: dict
            Contains source-specific parameters.
        spec_pars: dict
            Contains spectrum-specific parameters.
        
        """
        self._name = 'DiffuseSource'
        
        self.ion1 = evolve(src_pars['ion'])
        self.ion2 = evolve(src_pars['ion2'])
        self.heat = evolve(src_pars['heat'])        
        
    def SourceOn(self, t):
        return True
        
    def Luminosity(self, t=None):
        return 1.0
        
    def ionization_rate(self, t):
        return self.ion1(t)
    
    def secondary_ionization_rate(self, t):
        return self.ion2(t)
    
    def heating_rate(self, t):
        return self.heat(t)