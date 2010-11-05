""" 
Cosmology.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-03-01.

Description: Cosmology calculator based on Peebles 1993, with additions from Britton Smith's cosmology 
calculator in the Enzo analysis toolkit yt.

Notes: 
      -Everything here uses cgs.
      -I have assumed a flat universe for all calculations, i.e. OmegaCurvatureNow = 0.0.
      -WMAP VII cosmological parameters by default.

"""

import numpy as na
import pylab as pl
from Misc import *

c = 29979245800.0
G = 6.673*10**-8
km_per_mpc = 3.08568 * 10**13 * 10**6
cm_per_mpc = 3.08568 * 10**13 * 10**5 * 10**6
sqdeg_per_std = (180.0**2) / (na.pi**2)

class Cosmology:
    def __init__(self, pf):
        self.OmegaMatterNow = pf["OmegaMatterNow"]
        self.OmegaLambdaNow = pf["OmegaLambdaNow"]
        self.OmegaBaryonNow = pf["OmegaBaryonNow"]
        self.OmegaCDMNow = self.OmegaMatterNow - self.OmegaBaryonNow
        self.HubbleParameterNow = pf["HubbleParameterNow"] * 100 / km_per_mpc
        
        self.CriticalDensityNow = (3 * self.HubbleParameterNow**2) / (8 * na.pi * G)
        
    def LookbackTime(self, z_i, z_f):
        AgeIntegrand = lambda z: (1.0 / (z + 1.0) / self.EvolutionFunction(z))
        
        return (Romberg(AgeIntegrand, z_i, z_f) / self.HubbleParameterNow)    
        
    def ScaleFactor(self, z):
        return 1.0 / (1.0 + z)
        
    def EvolutionFunction(self, z):
        return na.sqrt(self.OmegaMatterNow * (1.0 + z)**3  + self.OmegaLambdaNow)
        
    def HubbleParameter(self, z):	
        return self.HubbleParameterNow * na.sqrt(self.OmegaMatterNow * (1.0 + z)**3 + 
            self.OmegaLambdaNow) 
    
    def OmegaMatter(self, z):
        return self.OmegaMatterNow * (1.0 + z)**3 / self.EvolutionFunction(z)**2
    
    def OmegaLambda(self, z):
	    return self.OmegaLambdaNow / self.EvolutionFunction(z)**2
    
    def MeanMatterDensity(self, z):
        return self.OmegaMatter(z) * self.CriticalDensity(z)
        
    def MeanBaryonDensity(self, z):
        return (self.OmegaBaryonNow / self.OmegaMatterNow) * self.MeanMatterDensity(z)
    
    def CriticalDensity(self, z):
        return (3.0 * self.HubbleParameter(z)**2) / (8.0 * na.pi * G)
        
            
    