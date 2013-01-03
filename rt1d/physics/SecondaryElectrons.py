"""
SecondaryElectrons.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-11-07.

Description: Read in Furlanetto & Stoever results, provide functions for 
interpolation of heating and ionization deposition fractions for fast 
secondary electrons.  Fits of Shull & vanSteenberg (1985) and Ricotti, 
Gnedin, & Shull (2002) also available.

"""

import h5py, os
import numpy as np

tiny_number = 1e-30

class SecondaryElectrons:
    def __init__(self, method = 0):
        self.Method = method

        if self.Method == 3:
            rt1d = os.environ.get("RT1D")
            if rt1d:
                f = h5py.File("%s/input/secondary_electron_data.h5" % rt1d, 'r')
            else:
                raise Exception('Error loading secondary electron data.')    
                
            # Read in Furlanetto & Stoever lookup tables
            self.E = f["electron_energy"].value
            self.x = f["ionized_fraction"].value
            
            from scipy.interpolate import RectBivariateSpline
            
            self.fh = RectBivariateSpline(self.E, self.x, f["f_heat"].value)
            self.fHI = RectBivariateSpline(self.E, self.x, f["fion_HI"].value)
            self.fHeI = RectBivariateSpline(self.E, self.x, f["fion_HeI"].value)
            self.fHeII = RectBivariateSpline(self.E, self.x, f["fion_HeII"].value)
            self.fexc = RectBivariateSpline(self.E, self.x, f["fexc"].value)
            self.flya = RectBivariateSpline(self.E, self.x, f['f_Lya'].value)
            
            f.close()        

    def DepositionFraction(self, E, xHII, channel = 'heat'):
        """
        Return the fraction of secondary electron energy deposited as heat, or further ionizations.
        The parameter 'channel' determines which we want, and could be:
        
            channel = (heat, h_1, he_1, he_2, lya)
        
        also,
                    
            Method = 0: OFF - all secondary electron energy goes to heat.
            Method = 1: Empirical fits of Shull & vanSteenberg 1985.
            Method = 2: Empirical Fits of Ricotti et al. 2002.
            Method = 3: Lookup tables of Furlanetto & Stoever 2010.
            
        """
        
        if E == 0.0: 
            E = tiny_number
        
        if self.Method == 0:
            if channel == 'heat':
                return np.ones_like(xHII)
            else: 
                return np.zeros_like(xHII)
            
        if self.Method == 1: 
            if channel == 'heat': 
                tmp = np.zeros_like(xHII)
                tmp[xHII <= 1e-4] = 0.15 * np.ones(len(tmp[xHII <= 1e-4]))
                tmp[xHII > 1e-4] = 0.9971 * (1. - pow(1 - 
                    pow(xHII[xHII > 1e-4], 0.2663), 1.3163))
                return tmp
            if channel == 'h_1': 
                return 0.3908 * pow(1. - pow(xHII, 0.4092), 1.7592)
            if channel == 'he_1': 
                return 0.0554 * pow(1. - pow(xHII, 0.4614), 1.6660) 
            if channel == 'he_2': 
                return np.zeros_like(xHII)
            if channel == 'lya': # Assuming that ALL excitations lead to a LyA photon
                return 0.4766 * pow(1. - pow(xHII, 0.2735), 1.5221)
            
        # Ricotti, Gnedin, & Shull (2002)
        if self.Method == 2:
            if channel == 'heat': 
                if xHII <= 1e-4: 
                    # This is what they do in Thomas & Zaroubi (2008).
                    return 0.15 * np.ones_like(xHII) 
                else: 
                    if E >= 11:
                        return 3.9811 * (11. / E)**0.7 * pow(xHII, 0.4) * \
                            (1. - pow(xHII, 0.34))**2 + \
                            (1. - (1. - pow(xHII, 0.2663))**1.3163)
                    else:
                        return np.ones_like(xHII)
                    
            if channel == 'h_1': 
                if E >= 28:
                    return -0.6941 * (28. / E)**0.4 * pow(xHII, 0.2) * \
                        (1. - pow(xHII, 0.38))**2 + \
                        0.3908 * (1. - pow(xHII, 0.4092))**1.7592
                else:
                    return np.zeros_like(xHII)
            if channel == 'he_1': 
                if E >= 28:
                    return -0.0984 * (28. / E)**0.4 * pow(xHII, 0.2) * \
                        (1. - pow(xHII, 0.38))**2 + \
                        0.0554 * (1. - pow(xHII, 0.4614))**1.6660
                else:
                    return np.zeros_like(xHII)
            if channel == 'he_2': 
                return np.zeros_like(xHII)
        
        # Furlanetto & Stoever (2010) - fix to handle array of xHII
        # Just set up a spline
        if self.Method == 3:
            
            f = np.zeros_like(xHII)
            
            for i, x in enumerate(xHII):
            
                if channel == 'heat': 
                    f[i] = self.fh(E, x)
                    #InterpolationTable = self.fheat
                if channel == 'h_1': 
                    f[i] = self.fHI(E, x)
                    #InterpolationTable = self.fion_HI
                if channel == 'he_1': 
                    InterpolationTable = self.fion_HeI
                if channel == 'he_2': 
                    InterpolationTable = self.fion_HeII
                if channel == 'lya':
                    InterpolationTable = self.f_Lya
            
            return f
            
