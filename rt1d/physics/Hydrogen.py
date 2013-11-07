"""

Hydrogen.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Mar 12 18:02:07 2012

Description: Container for hydrogen physics stuff.

"""

import numpy as np
from scipy.special import gamma
import scipy.interpolate as interpolate
from .Constants import A10, T_star, m_p, m_e, erg_per_ev, h, c, E_LyA, E_LL

try:
    from mathutils.differentiate import central_difference
except ImportError:
    pass

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

T_HH = np.array(T_HH)
T_He = np.array(T_He)

l_LyA = h * c / E_LyA / erg_per_ev

g23 = gamma(2./3.)
g13 = gamma(1./3.)

c1 = 4. * np.pi / 3. / np.sqrt(3.) / g23
c2 = 8. * np.pi / 3. / np.sqrt(3.) / g13

class Hydrogen:
    def __init__(self, cosm=None, approx_Salpha=1, approx_lya=0):
        if cosm is None:
            from .Cosmology import Cosmology
            self.cosm = Cosmology()
        else:
            self.cosm = cosm
            
        self.approx_S = approx_Salpha
        self.approx_lya = approx_lya
        
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
        
        self.kappa_H_pre = interpolate.interp1d(T_HH, kappa_HH, 
            kind='cubic', bounds_error=False, fill_value=0.0)
        self.kappa_e_pre = interpolate.interp1d(T_He, kappa_He, 
            kind='cubic', bounds_error=False, fill_value=0.0)
        
        self.tabulated_coeff = {'kappa_H': np.array(kappa_HH), 
                                'kappa_e': np.array(kappa_He), 
                                'T_H': np.array(T_HH), 'T_e': np.array(T_He)}
    
        # High-resolution tables for differentiation
        self._Tk_hi_H = np.logspace(np.log10(self.tabulated_coeff['T_H'].min()),
            np.log10(self.tabulated_coeff['T_H'].max()), 1000)
        self._kH_hi = np.array(map(self.kappa_H, self._Tk_hi_H))
        
        self._Tk_hi_e = np.logspace(np.log10(self.tabulated_coeff['T_e'].min()),
            np.log10(self.tabulated_coeff['T_e'].max()), 1000)
        self._ke_hi = np.array(map(self.kappa_e, self._Tk_hi_e))
                
        Tk_p_H, dkHdT = central_difference(self._Tk_hi_H, self._kH_hi)
        dlogkH_dlogT = dkHdT * Tk_p_H / np.array(map(self.kappa_H, Tk_p_H))
        
        _kH_spline = interpolate.interp1d(Tk_p_H, dlogkH_dlogT)
        self.dlogkH_dlogT = lambda T: _kH_spline(T)
        
        Tk_p_e, dkedT = central_difference(self._Tk_hi_e, self._ke_hi)
        dlogke_dlogT = dkedT * Tk_p_e / np.array(map(self.kappa_e, Tk_p_e))
        
        _ke_spline = interpolate.interp1d(Tk_p_e, dlogke_dlogT)
        self.dlogke_dlogT = lambda T: _ke_spline(T)
    
    def _kappa(self, Tk, Tarr, spline):
        if Tk < Tarr[0]:
            return spline(Tarr[0])
        elif Tk > Tarr[-1]:
            return spline(Tarr[-1])
        else:
            return spline(Tk)
                               
    def kappa_H(self, Tk):
        if type(Tk) in [float, np.float64]:            
            return self._kappa(Tk, T_HH, self.kappa_H_pre)
        else:
            tmp = np.zeros_like(Tk)
            for i in range(len(Tk)):
                tmp[i] = self._kappa(Tk[i], T_HH, self.kappa_H_pre)
            return tmp
            
    def kappa_e(self, Tk):                           
        if type(Tk) in [float, np.float64]:
            return self._kappa(Tk, T_He, self.kappa_e_pre)
        else:
            tmp = np.zeros_like(Tk)
            for i in range(len(Tk)):
                tmp[i] = self._kappa(Tk[i], T_He, self.kappa_e_pre)
            return tmp
            
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
        """ From Pritchard & Furlanetto 2006. """
        if n == 2:    return 1.0
        elif n == 3:  return 0.0
        elif n == 4:  return 0.2609
        elif n == 5:  return 0.3078
        elif n == 6:  return 0.3259
        elif n == 7:  return 0.3353
        elif n == 8:  return 0.3410
        elif n == 9:  return 0.3448
        elif n == 10: return 0.3476
        elif n == 11: return 0.3496
        elif n == 12: return 0.3512
        elif n == 13: return 0.3524
        elif n == 14: return 0.3535
        elif n == 15: return 0.3543
        elif n == 16: return 0.3550
        elif n == 17: return 0.3556
        elif n == 18: return 0.3561
        elif n == 19: return 0.3565
        elif n == 20: return 0.3569
        elif n == 21: return 0.3572
        elif n == 22: return 0.3575
        elif n == 23: return 0.3578
        elif n == 24: return 0.3580
        elif n == 25: return 0.3582
        elif n == 26: return 0.3584
        elif n == 27: return 0.3586
        elif n == 28: return 0.3587
        elif n == 29: return 0.3589
        elif n == 30: return 0.3590
        else:
            raise ValueError('Only know frec for 2 <= 2 <= 30!')
    
    # Look at line 905 in astrophysics.cc of jonathan's code
    
    def CollisionalCouplingCoefficient(self, Tk, z, nH, ne):
        """
        
        References
        ----------
        Zygelman, B. 2005, ApJ, 622, 1356
        """
        RateCoefficientSum = nH * self.kappa_H(Tk) + \
            ne * self.kappa_e(Tk)
                
        return RateCoefficientSum * T_star / A10 / self.cosm.TCMB(z)    
    
    def WouthuysenFieldCouplingCoefficient(self, z, Ja, Tk=None):
        """
        Return Lyman-alpha coupling coefficient.
        """
        
        return 1.81e11 * self.Sa(Tk) * Ja / (1. + z)
        
    def tauGP(self, z, xHII=0.):
        """ Gunn-Peterson optical depth. """
        return 1.5 * self.cosm.nH(z) * (1. - xHII) * l_LyA**3 * 50e6 \
            / self.cosm.HubbleParameter(z)
        
    def Sa(self, z=None, Tk=None, xHII=0.0):
        """
        Account for line profile effects.
        """
        
        if self.approx_S == 0:
            raise NotImplementedError('')
        if self.approx_S == 1:
            return 1.0
        elif self.approx_S == 2:
            return np.exp(-0.37 * np.sqrt(1. + z) * Tk**(-2./3.)) \
                / (1. + 0.4 / Tk)
        elif self.approx_S == 3:
            gamma = 1. / self.tauGP(z, xHII=xHII) / (1. + 0.4 / Tk)
            alpha = 0.717 * Tk**(-2./3.) * (1e-6 / gamma)**(1. / 3.)
            return 1. - c1 * alpha - c2 * alpha**2 + 4. * alpha**3 / 3.
        else:
            raise NotImplementedError('approx_Sa must be in [1,2,3].')
            
    def ELyn(self, n):
        """ Return energy of Lyman-n photon in eV. """
        return E_LL * (1. - 1. / n**2)
        
    def SpinTemp(self, z, Tk, Ja, nH, ne):
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
        x_a = self.WouthuysenFieldCouplingCoefficient(z, Ja, Tk)
        Tc = Tk
                
        return (1.0 + x_c + x_a) / \
            (self.cosm.TCMB(z)**-1. + x_c * Tk**-1. + x_a * Tc**-1.)
    
    def Ts(self, data, z):
        Tk = data['Tk']
        
        if self.approx_lya == 0:
            if 'Ja' in data.keys():
                Ja = data['Ja']
            else:
                Ja = np.zeros_like(Tk)
        elif self.approx_lya == 1:
            return Tk
        else:
            raise ValueError('approx_lya can only be 0 or 1!')
        
        nH = self.cosm.nH0 * (1. + z)**3
        ne = data['de']      
        
        return self.SpinTemp(z, Tk, Ja, nH, ne)
        
    def DifferentialBrightnessTemperature(self, z, xHII, delta, Ts):
        """
        Global 21-cm signature relative to cosmic microwave background in mK.
        """
        
        return 27. * (1. - xHII) * (1.0 + delta) * \
            (self.cosm.OmegaBaryonNow * self.cosm.h70**2 / 0.023) * \
            np.sqrt(0.15 * (1.0 + z) / self.cosm.OmegaMatterNow / self.cosm.h70**2 / 10.) * \
            (1.0 - self.cosm.TCMB(z) / Ts)
            

            