"""
Interpolate.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-10-30.

Description: Various interpolation routines
     
"""

import numpy as np

class Interpolate:
    def __init__(self, pf, n_col, itabs):
        self.pf = pf
        self.HIColumn = np.log10(n_col[0])
        self.HeIColumn = np.log10(n_col[1])
        self.HeIIColumn = np.log10(n_col[2])
        self.HINbins = len(self.HIColumn)
        self.HeINbins = len(self.HeIColumn)
        self.HeIINbins = len(self.HeIIColumn)
        self.HIColumnMin = self.HIColumn[0]
        self.HeIColumnMin = self.HeIColumn[0]
        self.HeIIColumnMin = self.HeIIColumn[0]
        self.dHIColumn = np.diff(self.HIColumn)[0]
        self.dHeIColumn = np.diff(self.HeIColumn)[0]
        self.dHeIIColumn = np.diff(self.HeIIColumn)[0]
        
        self.offsetHIColumn = self.HIColumnMin / self.dHIColumn
        self.offsetHeIColumn = self.HeIColumnMin / self.dHeIColumn
        self.offsetHeIIColumn = self.HeIIColumnMin / self.dHeIIColumn
        self.offsets = np.array([self.offsetHIColumn, self.offsetHeIColumn, self.offsetHeIIColumn])
        
        self.itabs = itabs  # This is a dictionary with all the lookup tables
        
        if self.pf["MultiSpecies"] == 0: self.interp = self.InterpolateLinear
        else: 
            if self.pf["InterpolationMethod"] == 0: self.interp = self.InterpolateTriLinear
            if self.pf["InterpolationMethod"] == 1: self.interp = self.InterpolateNN
                
    def InterpolateLinear(self, indices, integral, value = None):
        """
        Use this technique for hydrogen-only calculations.  For consistency with MultiSpecies > 0 methods, value 
        should still be a 3-element list.  
        """    
        
        return np.interp(np.log10(value[0]), self.HIColumn, self.itabs[integral])
        
    def InterpolateTriLinear(self, indices, integral, value = None):
        """
        Return the average of the 8 points surrounding the value of interest.
        """       
        
        ijk_s, ijk_b, xyz_d = indices  
        
        i_s, j_s, k_s = ijk_s 
        i_b, j_b, k_b = ijk_b
        x_d, y_d, z_d = xyz_d 
                
        i1 = self.itabs[integral][i_s][j_s][k_s] * (1 - z_d) + self.itabs[integral][i_s][j_s][k_b] * z_d
        i2 = self.itabs[integral][i_s][j_b][k_s] * (1 - z_d) + self.itabs[integral][i_s][j_b][k_b] * z_d
                                                                                              
        j1 = self.itabs[integral][i_b][j_s][k_s] * (1 - z_d) + self.itabs[integral][i_b][j_s][k_b] * z_d
        j2 = self.itabs[integral][i_b][j_b][k_s] * (1 - z_d) + self.itabs[integral][i_b][j_b][k_b] * z_d
        
        w1 = i1 * (1 - y_d) + i2 * y_d
        w2 = j1 * (1 - y_d) + j2 * y_d
                        
        return w1 * (1 - x_d) + w2 * x_d
    
    def GetIndices3D(self, value):
        """
        Retrieve set of 9 indices locating the interpolation points.
        
        value = 3-element array: [ncol_HI, ncol_HeI, ncol_HeII]
        """
        
        value = np.log10(value)
                                                
        # Smaller indices
        i_s = int((value[0] / self.dHIColumn) - self.offsetHIColumn)
        j_s = int((value[1] / self.dHeIColumn) - self.offsetHeIColumn)
        k_s = int((value[2] / self.dHeIIColumn) - self.offsetHeIIColumn)
        
        # Bracketing coordinates
        if i_s < 0: i_s = i_b = 0
        elif i_s >= (self.HINbins - 1): i_s = i_b = -1
        else: i_b = i_s + 1
        if j_s < 0: j_s = j_b = 0
        elif j_s >= (self.HeINbins - 1): j_s = j_b = -1
        else: j_b = j_s + 1
        if k_s < 0: k_s = k_b = 0
        elif k_s >= (self.HeIINbins - 1): k_s = k_b = -1
        else: k_b = k_s + 1        
                
        # Smaller values
        x_s = self.HIColumn[i_s]
        y_s = self.HeIColumn[j_s]
        z_s = self.HeIIColumn[k_s]
        
        # Bigger values
        x_b = self.HIColumn[i_b]
        y_b = self.HeIColumn[j_b]
        z_b = self.HeIIColumn[k_b]
                
        x_d = value[0] - x_s
        y_d = value[1] - y_s
        z_d = value[2] - z_s
        
        return [i_s, j_s, k_s], [i_b, j_b, k_b], [x_d, y_d, z_d]
        
    def InterpolateNN(self, indices, integral, value = None):
        """
        3D nearest neighbor interpolation.
        """  
        
        value = np.log10(np.array(value))
        
        # Analytically solve for positions in the array
        i = int(round((value[0] / self.dHIColumn) - self.offsetHIColumn))
        j = int(round((value[1] / self.dHeIColumn) - self.offsetHeIColumn))
        k = int(round((value[2] / self.dHeIIColumn) - self.offsetHeIIColumn))
        
        # Make sure we're still in the data cube
        i = min(self.HINbins - 1, max(0, i))
        j = min(self.HeINbins - 1, max(0, j))
        k = min(self.HeIINbins - 1, max(0, k))
                                                        
        return self.itabs[integral][i][j][k]
                  
        