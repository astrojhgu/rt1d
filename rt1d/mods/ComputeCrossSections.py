"""
ComputeCrossSections.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-10-25.

Description: Compute photo and collisional ionization cross sections as well as recombination rate 
coefficients and anything else like this that may crop up.
     
"""

import numpy as np

E_th = [13.6, 24.6, 54.4]
params = [[4.298e-1, 5.475e4, 3.288e1, 2.963, 0.0, 0.0, 0.0],
          [13.61, 9.492e2, 1.469, 3.188, 2.039, 4.434e-1, 2.136],
          [1.72, 1.369e4, 3.288e1, 2.963, 0.0, 0.0, 0.0]]

def PhotoIonizationCrossSection(E, species = 0):
    """ 
    Returns photoionization cross section for HI, HeI, or HeII from the fits of
    Verner et al. 1996.  HI is the first 7-element sub-array in 'params', HeI
    is the second, and HeII is the third.  In order, the coefficients in these arrays are:
        
        E_0, sigma_0, y_a, P, y_w, y_0, y_1
        
        Also:
        species = 0 for HI
        species = 1 for HeI
        species = 2 for HeII
        
        Note: The units are cm^2.
    
    """                            
    
    x = (E / params[species][0]) - params[species][5]
    y = np.sqrt(x**2 + params[species][6]**2)
    F_y = ((x - 1.0)**2 + params[species][4]**2) * \
        y**(0.5 * params[species][3] - 5.5) * \
        (1.0 + np.sqrt(y / params[species][2]))**-params[species][3]
                                
    return params[species][1] * F_y * 1e-18
                                                                                                               
        
    