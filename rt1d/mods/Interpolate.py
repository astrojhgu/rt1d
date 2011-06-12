"""
Interpolate3D.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-10-30.

Description: Various interpolation routines

Notes: 
     
"""

import numpy as np

def Interpolate1D(dataset, array, value, method = 0):
    """
    Interpolate a 1D dataset using one of the following methods:
    
        0: Linear
        1: Spline
        
    The variable 'dataset' is the 1D lookup table, 'array' contains
    the 1D array with the possible values for the independent variable.
    The input 'value' is the number we want a result for.
    
    """
    
    if len(array) == 3: array = array[0]
    if type(value) is not int: value = value[0]
        
    if method == 0: return np.interp(value, array, dataset)

def Interpolate3D(dataset, arrays, values, method = 0):
    """
    Interpolate a 3D dataset using one of the following methods:
    
        0: Nearest neighbors
        1: Average of 8 interpolation points
        2: Trilinear 
        3: Cubic spline
        4: crazy fancy interpolation algorithm
        
    The variable 'dataset' is the 3D lookup table, 'arrays' contains
    three 1D array with the possible values for the x, y, and z
    variables.  The input 'values' is a three element array containing
    the three values of x, y, and z we'd like to determine a result for.
    
    """
    
    xdiff = list(abs(arrays[0] - values[0]))
    ydiff = list(abs(arrays[1] - values[1]))
    zdiff = list(abs(arrays[2] - values[2]))
    
    imin1 = xdiff.index(min(xdiff))
    jmin1 = ydiff.index(min(ydiff))
    kmin1 = zdiff.index(min(zdiff))
        
    if imin1 == 0: imin2 = imin1 + 1
    elif imin1 == len(arrays[0]) - 1: imin2 = imin1 - 1
    else: imin2 = xdiff.index(min(xdiff[imin1 - 1], xdiff[imin1 + 1]))
    
    if jmin1 == 0: jmin2 = jmin1 + 1
    elif jmin1 == len(arrays[1]) - 1: jmin2 = jmin1 - 1
    else: jmin2 = ydiff.index(min(ydiff[jmin1 - 1], ydiff[jmin1 + 1]))
    
    if kmin1 == 0: kmin2 = kmin1 + 1
    elif kmin1 == len(arrays[2]) - 1: kmin2 = kmin1 - 1
    else: kmin2 = zdiff.index(min(zdiff[kmin1 - 1], zdiff[kmin1 + 1]))
        
    #print imin1, jmin1, kmin1    
        
    if method == 0: 
        return dataset[imin1][jmin1][kmin1]
    
    if method == 1:
        i1 = min(imin1, imin2)
        i2 = max(imin1, imin2)
        j1 = min(jmin1, jmin2)
        j2 = max(jmin1, jmin2)
        k1 = min(kmin1, kmin2)
        k2 = max(kmin1, kmin2)
        
        v1 = dataset[i1][j1][k1]
        v2 = dataset[i1][j1][k2]
        v3 = dataset[i1][j2][k1]
        v4 = dataset[i1][j2][k2]
        v5 = dataset[i2][j1][k1]
        v6 = dataset[i2][j1][k2]
        v7 = dataset[i2][j2][k1]
        v8 = dataset[i2][j2][k2]
                
        return np.mean([v1, v2, v3, v4, v5, v6, v7, v8])
        
    #if method == 2:
    #    return griddata(values, dataset, (arrays[0], arrays[1], arrays[2]), method='nearest')
        
        
        
        
        