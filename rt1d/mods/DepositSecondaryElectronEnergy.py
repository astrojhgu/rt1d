"""
DepositSecondaryElectronEnergy.py

Author: Jordan Mirocha (translated much of this from public Furlanetto & Stoever code)
Affiliation: University of Colorado at Boulder
Created on 2010-10-06.

Description: Read in newly formatted Furlanetto & Stoever results, interpolate to find
fraction of initial photon energy deposited in various channels.

Notes: -Can also calculate Shull & van Steenberg fit.
     
"""

import numpy as na

NumberOfEnergyBins = 258
NumberOfXiBins = 14

energies = na.zeros(NumberOfEnergyBins)
heat = na.zeros([NumberOfEnergyBins, NumberOfXiBins])
ionHI = na.zeros_like(heat)
ionHeI = na.zeros_like(heat)
ionHeII = na.zeros_like(heat)

x = [0.999, 0.990, 0.900, 0.500, 10**-1.0, 10**-1.3, 10**-1.6, 10**-2.0, 
     10**-2.3, 10**-2.6, 10**-3.0, 10**-3.3, 10**-3.6, 10**-4.0]
x.reverse()

def InterpolateDepositionFraction(energy, xi, channel = 'heat'):
    """
    Returns result from interpolation tables of Furlanetto & Stoever 2010.
    """        
    # Determine if E is within energy boundaries.  If not, set to closest boundary.
    if (energy > 0.999 * energies[NumberOfEnergyBins - 1]):
        energy = energies[NumberOfEnergyBins - 1] * 0.999
    elif(energy < energies[0]):
        energy = 0.0
        
    # Find lower index in energy table analytically.
    if (energy < 1008.88):
        i_low = int(na.log(energy / 10.0) / 1.98026273e-2)
    else:
        i_low = 232 + int(na.log(energy / 1008.88) / 9.53101798e-2)
      
    i_high = i_low + 1
    
    # Determine if ionized fraction is within energy boundaries.  If not, set to closest boundary.   
    if (xi > x[NumberOfXiBins - 1] * 0.999):
        xi = x[NumberOfXiBins - 1] * 0.999
    elif (xi < x[0]):
        xi = 1.001 * x[0];
    
    # Determine lower index in ionized fraction table iteratively.
    j_low = NumberOfXiBins - 1;
    while (xi < x[j_low]):
        j_low -= 1
    
    j_high = j_low + 1
    
    if channel == 'heat': table = heat
    elif channel == 'HI': table = ionHI
    elif channel == 'HeI': table = ionHeI
    elif channel == 'HeII': table = ionHeII
    
    # First linear interpolation in energy
    elow_result = ((table[i_high][j_low] - table[i_high][j_low]) / \
      	 (energies[i_high] - energies[i_low]))
    elow_result *= (energy - energies[i_low])
    elow_result += table[i_low][j_low]
  
    # Second linear interpolation in energy
    ehigh_result = ((table[i_high][j_high] - table[i_low][j_high]) / \
      	  (energies[i_high] - energies[i_low]))
    ehigh_result *= (energy - energies[i_low])
    ehigh_result += table[i_low][j_high]
    
    # Final interpolation over the ionized fraction
    final_result = (ehigh_result - elow_result) / (x[j_high] - x[j_low])
    final_result *= (xi - x[j_low])
    final_result += elow_result
  
    return final_result
    
def AnalyticDepositionFraction(xi, channel = 'heat'):    
    """Returns result from empirical formulae of Shull & van Steenberg 1985."""
    if channel == 'heat': return 0.9971 * (1 - pow(1 - pow(xi, 0.2663), 1.3163))
    if channel == 'HI': return 0.3908 * pow(1 - pow(xi, 0.4092), 1.7592)
    if channel == 'HeI': return 0.0554 * pow(1 - pow(xi, 0.4614), 1.6660)
    
def InitializeSecondaryElectronData():
    # Read in energies from table.
    f = open('secondary_electron_energies.dat', 'r')
    i = 0
    for line in f:
        if not line.strip(): continue
        if line.split()[0][0] == '#': continue
        
        energies[i] = float(line.split()[0])
        i += 1
        
    f.close()
    
    # Heating.
    f = open('secondary_electron_heat.dat', 'r')
    i = 0
    for line in f:
        if not line.strip(): continue
        if line.split()[0][0] == '#': continue
        
        for j in range(NumberOfXiBins):
          heat[i][j] = float(line.split()[j])
          
        i += 1
        
    f.close()
    
    # HI ionization.
    f = open('secondary_electron_ionHI.dat', 'r')
    i = 0
    for line in f:
        if not line.strip(): continue
        if line.split()[0][0] == '#': continue
        
        for j in range(NumberOfXiBins):
          ionHI[i][j] = float(line.split()[j])
          
        i += 1
        
    f.close()
    
    # HeI ionization.
    f = open('secondary_electron_ionHeI.dat', 'r')
    i = 0
    for line in f:
        if not line.strip(): continue
        if line.split()[0][0] == '#': continue
        
        for j in range(NumberOfXiBins):
          ionHeI[i][j] = float(line.split()[j])
          
        i += 1
        
    f.close()
    
    # HeII ionization.
    f = open('secondary_electron_ionHeI.dat', 'r')
    i = 0
    for line in f:
        if not line.strip(): continue
        if line.split()[0][0] == '#': continue
        
        for j in range(NumberOfXiBins):
          ionHeII[i][j] = float(line.split()[j])
          
        i += 1
        
    f.close()
    
InitializeSecondaryElectronData()