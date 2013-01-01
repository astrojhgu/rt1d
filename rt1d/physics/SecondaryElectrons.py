"""
SecondaryElectrons.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-11-07.

Description: Read in Furlanetto & Stoever results, provide functions for interpolation of heating ionization deposition
fractions for fast secondary electrons.  Fits of Shull & vanSteenberg (1985) and
Ricotti, Gnedin, & Shull (2002) also available.
     
Notes:
FJS10 logx_HII values interpolated first to grid where dlogx = const.     
     
To do: check in on i_low. 
     
"""

import h5py, os
import numpy as np
from ..util import parse_kwargs

tiny_number = 1e-30

# Furlanetto & Stoever ionized fractions (default)
x_HII = np.array([1.0e-4, 2.318e-4, 4.677e-4, 1.0e-3, 2.318e-3, 
                  4.677e-3, 1.0e-2, 2.318e-2, 4.677e-2, 1.0e-1, 
                  0.5, 0.9, 0.99, 0.999])

class SecondaryElectrons:
    def __init__(self, **kwargs):
        self.pf = parse_kwargs(**kwargs)
        self.Method = self.pf['SecondaryIonization']
        self.NumberOfEnergyBins = 258
        self.NumberOfXiBins = 14
        
        self.log_xHII = np.linspace(np.log10(min(x_HII)), np.log10(max(x_HII)),
            self.pf['IonizedFractionBins'])
        
        if self.Method >= 2:
            rt1d = os.environ.get("RT1D")
            if rt1d:
                f = h5py.File("{0}/input/secondary_electron_data.h5".format(rt1d), 'r')
            elif pf["SecondaryElectronDataFile"] is not 'None':
                f = h5py.File("%s" % pf["SecondaryElectronDataFile"], 'r')
            else:
                raise Exception('Error loading secondary electron data.')
                
            # Read in Furlanetto & Stoever lookup tables
            self.Energies = f["electron_energy"].value
            self.IonizedFractions = f["ionized_fraction"].value
            self.LogIonizedFractions = np.log10(self.IonizedFractions)    
            self.fheat = f["f_heat"].value
            self.fion_HI = f["fion_HI"].value
            self.fion_HeI = f["fion_HeI"].value
            self.fion_HeII = f["fion_HeII"].value
            self.fexc = f['fexc'].value
            self.f_Lya = f['f_Lya'].value
            self.fion = f['fion'].value
            
            self.xmin = min(self.IonizedFractions)
            self.xmax = max(self.IonizedFractions)
            
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
            
            if channel == 'heat': 
                InterpolationTable = self.fheat
            if channel == 'h_1': 
                InterpolationTable = self.fion_HI
            if channel == 'he_1': 
                InterpolationTable = self.fion_HeI
            if channel == 'he_2': 
                InterpolationTable = self.fion_HeII
            if channel == 'lya':
                InterpolationTable = self.f_Lya
            
            # Determine if E is within energy boundaries.  If not, set to closest boundary.
            if (E > 0.999 * self.Energies[self.NumberOfEnergyBins - 1]):
                E = self.Energies[self.NumberOfEnergyBins - 1] * 0.999
            elif(E < self.Energies[0]):
                if channel == 0:
                    return np.ones_like(xHII)
                else:
                    return np.zeros_like(xHII)
                
            # Find lower index in energy table analytically.
            if (E < 1008.88):
                i_low = int(np.log(E / 10.0) / 1.98026273e-2)
            else:
                i_low = 232 + int(np.log(E / 1008.88) / 9.53101798e-2)
              
            i_high = i_low + 1
                        
            # Determine if ionized fraction is within energy boundaries.  If not, set to closest boundary.   
            if (xHII > self.IonizedFractions[self.NumberOfXiBins - 1] * 0.9999999):
                xHII = self.IonizedFractions[self.NumberOfXiBins - 1] * 0.9999999
            elif (xHII < self.IonizedFractions[0]):
                xHII = 1.0000001 * self.IonizedFractions[0];
            
            # Determine lower index in ionized fraction table iteratively.
            j_low = self.NumberOfXiBins - 1;
            while (xHII < self.IonizedFractions[j_low]):
                j_low -= 1
            
            j_high = j_low + 1
            
            # First linear interpolation in energy
            elow_result = ((InterpolationTable[i_high][j_low] - InterpolationTable[i_high][j_low]) / \
              	 (self.Energies[i_high] - self.Energies[i_low]))
            elow_result *= (E - self.Energies[i_low])
            elow_result += InterpolationTable[i_low][j_low]
            
            # Second linear interpolation in energy
            ehigh_result = ((InterpolationTable[i_high][j_high] - InterpolationTable[i_low][j_high]) / \
              	  (self.Energies[i_high] - self.Energies[i_low]))
            ehigh_result *= (E - self.Energies[i_low])
            ehigh_result += InterpolationTable[i_low][j_high]
            
            # Final interpolation over the ionized fraction
            final_result = (ehigh_result - elow_result) / (self.IonizedFractions[j_high] - self.IonizedFractions[j_low])
            final_result *= (xHII - self.IonizedFractions[j_low])
            final_result += elow_result
            
            return final_result
                
            