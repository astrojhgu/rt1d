"""
Interpolate3D.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-10-30.

Description: Various interpolation routines

Notes: 
     
"""

import numpy as np

class Interpolate:
    def __init__(self, pf, n_col, itabs):
        self.pf = pf
        self.HIColumn = n_col[0]
        self.HeIColumn = n_col[1]
        self.HeIIColumn = n_col[2]
        self.HINbins = len(self.HIColumn)
        self.HeINbins = len(self.HeIColumn)
        self.HeIINbins = len(self.HeIIColumn)
        self.HIColumnMin = np.log10(self.HIColumn[0])
        self.HeIColumnMin = np.log10(self.HeIColumn[0])
        self.HeIIColumnMin = np.log10(self.HeIIColumn[0])
        self.dHIColumn = np.diff(np.log10(self.HIColumn))[0]
        self.dHeIColumn = np.diff(np.log10(self.HeIColumn))[0]
        self.dHeIIColumn = np.diff(np.log10(self.HeIIColumn))[0]
        
        self.itabs = itabs  # This is a dictionary with all the lookup tables
        
        if self.pf["MultiSpecies"] == 0: self.interp = self.InterpolateLinear1D
        else: 
            if self.pf["InterpolationMethod"] == 0: self.interp = self.InterpolateNN3D
            if self.pf["InterpolationMethod"] == 1: self.interp = self.InterpolateAvg3D
                
    def InterpolateLinear1D(self, value, integral):
        """
        Use this technique for hydrogen-only calculations.  For consistency with MultiSpecies > 0 methods, value 
        should still be a 3-element list.  
        """    
                                
        return np.interp(value[0], self.HIColumn, self.itabs[integral])
        
    def InterpolateNN3D(self, value, integral):
        """
        3D nearest neighbor interpolation.
        """  
        
        i = int(round((np.log10(value[0]) - self.HIColumnMin) / self.dHIColumn))  
        i = min(self.HINbins - 1, max(0, i))
        j = int(round((np.log10(value[1]) - self.HeIColumnMin) / self.dHeIColumn))
        j = min(self.HeINbins - 1, max(0, i))
        k = int(round((np.log10(value[2]) - self.HeIIColumnMin) / self.dHeIIColumn))
        k = min(self.HeIINbins - 1, max(0, i))
        
        return self.itabs[integral][i][j][k]
        
    def InterpolateAvg3D(self, value, integral):
        """
        Return the average of the 8 points surrounding the value of interest.
        """    
        pass
        
    
    
            
        
        
        
        