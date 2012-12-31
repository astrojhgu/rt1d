"""

Hydrogen.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Mar 12 18:02:07 2012

Description: Container for hydrogen physics stuff.

"""

import numpy as np
import scipy.interpolate as interpolate
from .Constants import A10, T_star, m_p, m_e, erg_per_ev, h, c

# Rate coefficients for spin de-excitation - from Zygelman originally

# H-H collisions.
T_HH = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0,
        90.0, 100.0, 200.0, 300.0, 500.0, 700.0, 1000.0, 2000.0, 3000.0, 5000.0, 7000.0, 10000.0]

kappa_HH = [1.38e-13, 1.43e-13, 2.71e-13, 6.60e-13, 1.47e-12, 2.88e-12, 9.10e-12, 1.78e-11, 2.73e-11,
            3.67e-11, 5.38e-11, 6.86e-11, 8.14e-11, 9.25e-11, 1.02e-10, 1.11e-10, 1.19e-10, 1.75e-10,
            2.09e-10, 2.56e-10, 2.91e-10, 3.31e-10, 4.27e-10, 4.97e-10, 6.03e-10, 6.87e-10, 7.87e-10]
            
# H-e collisions.            
T_He = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 3000.0, 5000.0, 7000.0, 
        10000.0, 15000.0, 20000.0]
        
kappa_He = [2.39e-10, 3.37e-10, 5.30e-10, 7.46e-10, 1.05e-9, 1.63e-9, 2.26e-9, 3.11e-9, 4.59e-9, 5.92e-9,
            7.15e-9, 7.71e-9, 8.17e-9, 8.32e-9, 8.37e-9, 8.29e-9, 8.11e-9]

class Hydrogen:
    def __init__(self, cosm = None):
        if cosm is None:
            from .Cosmology import Cosmology
            self.cosm = Cosmology()
        else:
            self.cosm = cosm
        
        self.nmax = 23
        self.fbarII = 0.72
        self.fbarIII = 0.63
        self.A10 = 2.85e-15 			
        self.E10 = 5.9e-6 				
        self.m_H = m_p + m_e     		
        self.nu_0 = 1420.4057e6 			
        self.T_star = 0.068 				
        self.a_0 = 5.292e-9 				
                
        # Common lines, etc.
        self.nu_LL = 13.6 * erg_per_ev / h
        self.E_LyA = h * c / (1216. * 1e2 / 1e10) / erg_per_ev
        self.E_LyB = h * c / (1026. * 1e2 / 1e10) / erg_per_ev
        self.E_LL = h * self.nu_LL / erg_per_ev
        self.nu_alpha = self.E_LyA * erg_per_ev / h
        self.nu_beta = self.E_LyB * erg_per_ev / h
        self.dnu = self.nu_LL - self.nu_alpha #(13.6 - 11.18) * erg_per_ev / h  
        
        self.kappa_H = interpolate.interp1d(T_HH, kappa_HH, 
            kind = 'cubic', bounds_error = False, fill_value = 0.0)
        self.kappa_e = interpolate.interp1d(T_He, kappa_He, 
            kind = 'cubic', bounds_error = False, fill_value = 0.0)
        
        self.tabulated_coeff = {'kappa_H': kappa_HH, 'T_H': T_HH, 
                                'kappa_e': kappa_He, 'T_e': T_He}
                                
    def CollisionalIonizationRate(self, T):
        """
        From Fukugita & Kawasaki 1996 (I think).
        """
        return 5.85e-11 * np.sqrt(T) * (1. + np.sqrt(T / 1.e5))**-1. * np.exp(-1.578e5 / T)    
                
    def RecombinationRateCaseA(self, Tk):
        """
        Return mean IGM recombination rate coefficient.
        """    
        
        return 4.2e-13 * (Tk / 1.e4)**-0.7    
    
    def RecombinationRateCaseB(self, Tk):
        """
        Return mean IGM recombination rate coefficient.
        """    
        
        return 2.6e-13 * (Tk / 1.e4)**-0.7                            

    def photon_energy(self, nu, nl = 1):
        """
        Return energy of photon transitioning from nu to nl in eV.  
        Defaults to Lyman-series.
        """
        return Ryd * (1. / nl / nl - 1. / nu / nu) / erg_per_ev

    def photon_freq(self, nu, nl = 1):
        return self.photon_energy(nu, nl) * erg_per_ev / h 

    def zmax(self, z, n):
        return (1. + z) * (1. - (n + 1)**-2) / (1. - n**-2) - 1.
    
    def OnePhotonPerHatom(self, z):
        """
        Flux of photons = 1 photon per hydrogen atom assuming Lyman alpha 
        frequency.
        """
        
        return self.cosm.nH0 * (1. + z)**3 * c / 4. / np.pi / self.nu_alpha
    
    def frec(self, n):
        if n == 2: return 1.0
        if n == 3: return 0.0
        if n == 4: return 0.2609
        if n == 5: return 0.3078
        if n == 6: return 0.3259
        if n == 7: return 0.3353
        if n == 8: return 0.3410
        if n == 9: return 0.3448
        if n == 10: return 0.3476
        if n == 11: return 0.3496
        if n == 12: return 0.3512
        if n == 13: return 0.3524
        if n == 14: return 0.3535
        if n == 15: return 0.3543
        if n == 16: return 0.3550
        if n == 17: return 0.3556
        if n == 18: return 0.3561
        if n == 19: return 0.3565
        if n == 20: return 0.3569
        if n == 21: return 0.3572
        if n == 22: return 0.3575
        if n == 23: return 0.3578
        if n == 24: return 0.3580
        if n == 25: return 0.3582
        if n == 26: return 0.3584
        if n == 27: return 0.3586
        if n == 28: return 0.3587
        if n == 29: return 0.3589    
        if n == 30: return 0.3590
    
    # Look at line 905 in astrophysics.cc of jonathan's code
    
    def CollisionalCouplingCoefficient(self, Tk, z, nH, ne):
        RateCoefficientSum = nH * self.kappa_H(Tk) + \
            ne * self.kappa_e(Tk)
                
        return RateCoefficientSum * T_star / A10 / self.cosm.TCMB(z)    
    
    def WouthuysenFieldCouplingCoefficient(self, z, Ja):
        """
        Return Lyman-alpha coupling coefficient.
        """
        
        Sa = 1. # for now
        
        return 1.81e11 * Sa * Ja / (1. + z)
    
    def Ts(self, z, Tk, Ja, nH, ne):
        """
        Returns spin temperature given:
            z = redshift (sets CMB temperature)
            Tk = kinetic temperature of the gas
            xHII = ionized hydrogen fraction
            Ja = Lyman-alpha flux
            nH = proper hydrogen density
            ne = electron density
        """

        x_c = self.CollisionalCouplingCoefficient(Tk, z, nH, ne)
        x_a = self.WouthuysenFieldCouplingCoefficient(z, Ja)
        Tc = Tk # for now
                
        return (1.0 + x_c + x_a) / \
            (self.cosm.TCMB(z)**-1. + x_c * Tk**-1. + x_a * Tc**-1.)
        
    def DifferentialBrightnessTemperature(self, z, xHII, delta, Ts):
        """
        Global 21-cm signature relative to cosmic microwave background in mK.
        """
        
        return 27. * (1. - xHII) * (1.0 + delta) * \
            (self.cosm.OmegaBaryonNow * self.cosm.h70**2 / 0.023) * \
            np.sqrt(0.15 * (1.0 + z) / self.cosm.OmegaMatterNow / self.cosm.h70**2 / 10.) * \
            (1.0 - self.cosm.TCMB(z) / Ts)
            
    def AbsorptionSignal(self, z, Tk, Ja):
        """
        Signal assuming neutral medium.
        """        
        
        nH = self.cosm.nH0 * (1. + z)**3
        Ts = self.Ts(z, Tk, Ja, nH, 0.0)
        
        return 27. * \
            (self.cosm.OmegaBaryonNow * self.cosm.h70**2 / 0.023) * \
            np.sqrt(0.15 * (1.0 + z) / self.cosm.OmegaMatterNow / self.cosm.h70**2 / 10.) * \
            (1.0 - self.cosm.TCMB(z) / Ts)
            
            
            
            