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

import numpy as np
from .Constants import c, G, km_per_mpc, m_H
from .Integrate import simpson

class Cosmology:
    def __init__(self, pf):
        self.pf = pf
        self.OmegaMatterNow = pf["OmegaMatterNow"]
        self.OmegaLambdaNow = pf["OmegaLambdaNow"]
        self.OmegaBaryonNow = pf["OmegaBaryonNow"]
        self.OmegaCDMNow = self.OmegaMatterNow - self.OmegaBaryonNow
        self.HubbleParameterNow = pf["HubbleParameterNow"] * 100 / km_per_mpc
        
        self.CriticalDensityNow = (3 * self.HubbleParameterNow**2) / (8 * np.pi * G)
        
        self.Y = self.pf["PrimordialHeliumByMass"] * self.pf["MultiSpecies"]            
        self.y = self.Y / 4. / (1. - self.Y) 
        
        self.X = 1. - self.Y

        # Hydrogen, helium, electron, and baryon densities today (z = 0)
        self.rho_b_z0 = self.MeanBaryonDensity(0)
        self.nH0 = (1. - self.Y) * self.rho_b_z0 / m_H
        self.nHe0 = self.y * self.nH0
        self.ne0 = self.nH0 + 2. * self.nHe0
        self.rho_n_z0 = self.nH0 + self.nHe0 + self.ne0
        
    def TimeToRedshiftConverter(self, t_i, t_f, z_i):
        """
        High redshift approximation under effect.
        """
        return ((1. + z_i)**(-3. / 2.) + (3. * self.HubbleParameterNow * np.sqrt(self.OmegaMatterNow) * (t_f - t_i) / 2.))**(-2. / 3.) - 1.
        
    def LookbackTime(self, z_i, z_f):
        """
        Returns lookback time from z_i to z_f in seconds, where z_i < z_f.
        """
        return 2. * ((1. + z_i)**-1.5 - (1. + z_f)**-1.5) / \
            np.sqrt(self.OmegaMatterNow) / self.HubbleParameterNow / 3.    
        
    def TCMB(self, z):
        return 2.725 * (1. + z)    
        
    def ScaleFactor(self, z):
        return 1.0 / (1.0 + z)
        
    def EvolutionFunction(self, z):
        return np.sqrt(self.OmegaMatterNow * (1.0 + z)**3  + self.OmegaLambdaNow)
        
    def HubbleParameter(self, z):	
        return self.HubbleParameterNow * np.sqrt(self.OmegaMatterNow * (1.0 + z)**3 + 
            self.OmegaLambdaNow) 
    
    def OmegaMatter(self, z):
        return self.OmegaMatterNow * (1.0 + z)**3 / self.EvolutionFunction(z)**2
    
    def OmegaLambda(self, z):
	    return self.OmegaLambdaNow / self.EvolutionFunction(z)**2
    
    def MeanMatterDensity(self, z):
        return self.OmegaMatter(z) * self.CriticalDensity(z)
        
    def MeanBaryonDensity(self, z):
        return (self.OmegaBaryonNow / self.OmegaMatterNow) * self.MeanMatterDensity(z)
    
    def MeanHydrogenNumberDensity(self, z):
        return (1. - self.Y) * self.MeanBaryonDensity(z) / m_H
        
    def MeanHeliumNumberDensity(self, z):
        return self.Y * self.MeanBaryonDensity(z) / m_He    
    
    def CriticalDensity(self, z):
        return (3.0 * self.HubbleParameter(z)**2) / (8.0 * np.pi * G)
    
    def dtdz(self, z):
        return 1. / self.HubbleParameter(z) / (1. + z) 
    
            
    