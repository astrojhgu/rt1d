""" 
SpinTemperature.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-05-25.

Description: Calculate HI spin temperature using collisional de-excitation
rate coefficients from Furlanetto et al. 2006, tables 3-4.

Notes: 
"""

import numpy as np
from .Cosmology import Cosmology
from .Constants import A10, T_star, m_p, nu_alpha, f12, e, m_e, c, h

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

class SpinTemperature:
    def __init__(self, pf):
        self.pf = pf
        self.cosm = Cosmology(pf)
        
    def _CollisionalCouplingCoefficient(self, z, n_H, n_e, x_HII, Tk):
        """
        From Furlanetto et al. 2006, tables 3-4.
        """
        C10 = n_H * (1. - x_HII) * np.interp(Tk, T_HH, kappa_HH) + \
            n_e * np.interp(Tk, T_He, kappa_He)
                                          
        return C10 * T_star / A10 / self.cosm.TCMB(z)
    
    def _WouthuysenFieldCouplingCoefficient(self, z, n_H, x_HII, Tk, Jc, Ji):
        """
        Return Lyman-alpha coupling coefficient.
        """
        
        H = self.cosm.HubbleParameter(z)
        
        Sc = Si = 1.
        
        # Fiducial Jalpha (energy units)
        J0 = (h * nu_alpha) * c * n_H / 4. / np.pi / nu_alpha
        
        # Gunn-Peterson optical depth
        tau_GP = (1. - x_HII) * np.pi * (e * c / 10.)**2 * n_H * f12 \
            / H / m_e / nu_alpha
        
        P10 = (4. / 27.) * H * tau_GP * (Sc * Jc + Si * Ji) / J0
                
        return P10 * T_star / A10 / self.cosm.TCMB(z)
            
    def Ts(self, z, n_H, n_e, Tk, x_HII, Jc, Ji):
        """
        Returns spin temperature given:
            z = redshift (sets CMB temperature)
            Tk = kinetic temperature of the gas
            x_e = electron fraction
            x_H = neutral hydrogen fraction
            Jc, Ji = Lyman-alpha flux (energy units)
        """

        x_c = self._CollisionalCouplingCoefficient(z, n_H, n_e, x_HII, Tk)
        x_a = self._WouthuysenFieldCouplingCoefficient(z, n_H, x_HII, Tk, Jc, Ji)
        Tc = Tk # for now
                
        return (1. + x_c + x_a) / (self.cosm.TCMB(z)**-1. + x_c / Tk + x_a / Tc)
                        
        