"""
SecondaryElectrons.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-11-07.

Description: Read in Furlanetto & Stoever results, provide functions for interpolation of heating ionization deposition
fractions for fast secondary electrons.  Fits of Shull & vanSteenberg also available.
     
"""

import h5py, os
import numpy as np

class SecondaryElectrons:
    def __init__(self, pf):
        self.pf = pf
        self.Method = self.pf["SecondaryElectronMethod"]
        self.NumberOfEnergyBins = 258
        self.NumberOfXiBins = 14
        
        rt1d = os.environ.get("RT1D")
        f = h5py.File("{0}/input/secondary_electron_data.h5".format(rt1d), 'r')
        
        # Read in Furlanetto & Stoever lookup tables
        self.Energies = f["ElectronEnergy"].value
        self.IonizedFractions = f["IonizedFraction"].value
        self.Heat = f["Heat"].value
        self.IonizationHI = f["IonizationHI"].value
        self.IonizationHeI = f["IonizationHeI"].value
        self.IonizationHeII = f["IonizationHeII"].value
        
        f.close()
        
    def DepositionFraction(self, E, xi, channel = 0):
        """
        Return the fraction of secondary electron energy deposited as heat, or further ionizations.
        The parameter 'channel' determines which we want, with:
        
            channel = 0: heat
            channel = 1: ionization of HI
            channel = 2: ionization of HeI
            channel = 3: ionization of HeII
            
        and
        
            Method = 0: OFF - ignoring the effects of secondary electrons
            Method = 1: Empirical fits of Shull & vanSteenberg 1985
            Method = 2: Lookup tables of Furlanetto & Stoever 2010
        
        """
        
        if self.Method == 0:
            return 0.0
            
        if self.Method == 1:
            
            if channel == 0: InterpolationTable = self.Heat
            if channel == 1: InterpolationTable = self.IonizationHI
            if channel == 2: InterpolationTable = self.IonizationHeI
            if channel == 3: InterpolationTable = self.IonizationHeII
            
            # Determine if E is within energy boundaries.  If not, set to closest boundary.
            if (E > 0.999 * self.Energies[self.NumberOfEnergyBins - 1]):
                E = self.Energies[self.NumberOfEnergyBins - 1] * 0.999
            elif(E < self.Energies[0]):
                E = 0.0
                
            # Find lower index in energy table analytically.
            if (E < 1008.88):
                i_low = int(np.log(E / 10.0) / 1.98026273e-2)
            else:
                i_low = 232 + int(np.log(E / 1008.88) / 9.53101798e-2)
              
            i_high = i_low + 1
            
            # Determine if ionized fraction is within energy boundaries.  If not, set to closest boundary.   
            if (xi > self.IonizedFractions[self.NumberOfXiBins - 1] * 0.999):
                xi = self.IonizedFractions[self.NumberOfXiBins - 1] * 0.999
            elif (xi < self.IonizedFractions[0]):
                xi = 1.001 * self.IonizedFractions[0];
            
            # Determine lower index in ionized fraction table iteratively.
            j_low = self.NumberOfXiBins - 1;
            while (xi < self.IonizedFractions[j_low]):
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
            final_result *= (xi - self.IonizedFractions[j_low])
            final_result += elow_result
            
            return final_result
            
        if self.Method == 2:
            if channel == 0: return 0.9971 * (1 - pow(1 - pow(xi, 0.2663), 1.3163))
            if channel == 1: return 0.3908 * pow(1 - pow(xi, 0.4092), 1.7592)
            if channel == 2: return 0.0554 * pow(1 - pow(xi, 0.4614), 1.6660)    
            
            
            