"""
Interpolate.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-10-30.

Description: Various interpolation routines
     
"""

import numpy as np
from .SecondaryElectrons import SecondaryElectrons
from scipy.interpolate import LinearNDInterpolator

class Interpolate:
    def __init__(self, pf, n_col, itabs):
        self.pf = pf
        self.esec = SecondaryElectrons(pf)
        
        self.MultiSpecies = self.pf["MultiSpecies"]
        
        self.HIColumn = n_col[0]
        self.HeIColumn = n_col[1]
        self.HeIIColumn = n_col[2]
        self.HINbins = len(self.HIColumn)
        self.HeINbins = len(self.HeIColumn)
        self.HeIINbins = len(self.HeIIColumn)
        self.HIColumnMin = self.HIColumn[0]
        self.HeIColumnMin = self.HeIColumn[0]
        self.HeIIColumnMin = self.HeIIColumn[0]
        self.HIColumnMax = self.HIColumn[-1]
        self.HeIColumnMax = self.HeIColumn[-1]
        self.HeIIColumnMax = self.HeIIColumn[-1]
        self.dHIColumn = np.diff(self.HIColumn)[0]
        self.dHeIColumn = np.diff(self.HeIColumn)[0]
        self.dHeIIColumn = np.diff(self.HeIIColumn)[0]
                                        
        self.offsetHIColumn = self.HIColumnMin / self.dHIColumn
        self.offsetHeIColumn = self.HeIColumnMin / self.dHeIColumn
        self.offsetHeIIColumn = self.HeIIColumnMin / self.dHeIIColumn
        
        self.AllColumns = [self.HIColumn, self.HeIColumn, self.HeIIColumn]
        
        # This is a dictionary with all the lookup tables
        self.itabs = itabs
        
        # What kind of interpolator do we need?
        if self.MultiSpecies == 0: 
            if self.esec.Method < 2:
                self.interp = self.InterpolateLinear
                self.GetIndices = self.GetIndices1D
            else:
                self.interp = self.InterpolateBiLinear
                self.GetIndices = self.GetIndices2D
        else: 
            if self.esec.Method < 2:
                self.interp = self.InterpolateTriLinear
                self.GetIndices = self.GetIndices3D
            else:
                self.interp = self.InterpolateQuadLinear 
                self.GetIndices = self.GetIndices4D
            
    # Do we still need this anywhere?        
    def OpticalDepth(self, value, species):
        return 10**np.interp(value, self.AllColumns[species], self.itabs['OpticalDepth%i' % species])
                
    def InterpolateLinear(self, indices, integral, value, x_HII = None):
        """
        Use this technique for hydrogen-only calculations.  For consistency with MultiSpecies > 0 
        methods, value should still be a 3-element list.  
        """    
                                
        return 10**np.interp(value[0], self.HIColumn, self.itabs[integral])
        
    def InterpolateBiLinear(self, indices, integral, value, x_HII = None):
        """
        We use this for runs when MultiSpecies = 0 and SecondaryIonization = 2
        (with DiscreteSpectrum = 0, TabulateIntegrals = 1).
        """    
        
        if x_HII is None:
            return self.InterpolateLinear(indices, integral, value)
        
        i_n, i_x = indices      
                                                
        x1 = self.HIColumn[i_n]
        x2 = self.HIColumn[i_n + 1]
        y1 = self.esec.LogIonizedFractions[i_x]
        y2 = self.esec.LogIonizedFractions[i_x + 1]
        
        f11 = self.itabs[integral][i_n][i_x]
        f21 = self.itabs[integral][i_n + 1][i_x]
        f12 = self.itabs[integral][i_n][i_x + 1]
        f22 = self.itabs[integral][i_n + 1][i_x + 1]
                                
        final = (f11 * (x2 - value[0]) * (y2 - x_HII) + \
            f21 * (value[0] - x1) * (y2 - x_HII) + \
            f12 * (x2 - value[0]) * (x_HII - y1) + \
            f22 * (value[0] - x1) * (x_HII - y1)) / (x2 - x1) / (y2 - y1)    
                        
        return 10**final    
        
    def InterpolateTriLinear(self, indices, integral, value = None, x_HII = None):
        """
        Return the average of the 8 points surrounding the value of interest.
        """       
                
        ijk_s, ijk_b, xyz_d = indices  
        
        i_s, j_s, k_s = ijk_s 
        i_b, j_b, k_b = ijk_b
        x_d, y_d, z_d = xyz_d         
                
        i1 = self.itabs[integral][i_s][j_s][k_s] * (1. - z_d) + self.itabs[integral][i_s][j_s][k_b] * z_d
        i2 = self.itabs[integral][i_s][j_b][k_s] * (1. - z_d) + self.itabs[integral][i_s][j_b][k_b] * z_d
                                                                                              
        j1 = self.itabs[integral][i_b][j_s][k_s] * (1. - z_d) + self.itabs[integral][i_b][j_s][k_b] * z_d
        j2 = self.itabs[integral][i_b][j_b][k_s] * (1. - z_d) + self.itabs[integral][i_b][j_b][k_b] * z_d
        
        w1 = i1 * (1. - y_d) + i2 * y_d
        w2 = j1 * (1. - y_d) + j2 * y_d
                                                                                
        final = w1 * (1. - x_d) + w2 * x_d
        
        return 10**final
    
    def InterpolateQuadLinear(self, indices, integral, value, x_HII = None):
        """
        This gets called if MultiSpecies = 1 and SecondaryIonization = 2
        (and DiscreteSpectrum = 0, TabulateIntegrals = 1).
        """    
        
        if x_HII is None:
            return self.InterpolateTriLinear(indices, integral, value)
        else:
            pass
            
            # Do stuff
            
    def GetIndices1D(self, value = None, x_HII = None):
        return None        
            
    def GetIndices2D(self, value, x_HII):
        """
        Return column density and ionized fraction indices.
        """    
                
        i_nHI = int((value[0] - self.HIColumnMin) / self.dHIColumn)
        
        if x_HII > self.esec.LogIonizedFractions[-1]:
            i_xHII = self.esec.NumberOfXiBins - 2
        elif x_HII <= self.esec.LogIonizedFractions[0]:
            i_xHII = 0
        else:    
            # Determine lower index in ionized fraction table iteratively.
            i_xHII = self.esec.NumberOfXiBins - 1
            while (x_HII < self.esec.LogIonizedFractions[i_xHII]):
                i_xHII -= 1
                
        return max(i_nHI, 0), i_xHII
        
    def GetIndices3D(self, value, x_HII = None):
        """
        Retrieve set of 9 indices locating the interpolation points.
        
        value = 3-element array: [ncol_HI, ncol_HeI, ncol_HeII]
        """
        
        if not self.MultiSpecies:
            return None
                                                        
        # Smaller indices
        i_s = int((value[0] / self.dHIColumn) - self.offsetHIColumn)
        j_s = int((value[1] / self.dHeIColumn) - self.offsetHeIColumn)
        k_s = int((value[2] / self.dHeIIColumn) - self.offsetHeIIColumn)
        
        # Bracketing coordinates
        if i_s < 0: 
            i_s = i_b = 0
        elif i_s >= (self.HINbins - 1): 
            i_s = i_b = -1
        else: 
            i_b = i_s + 1
        if j_s < 0: 
            j_s = j_b = 0
        elif j_s >= (self.HeINbins - 1): 
            j_s = j_b = -1
        else: 
            j_b = j_s + 1
        if k_s < 0: 
            k_s = k_b = 0
        elif k_s >= (self.HeIINbins - 1): 
            k_s = k_b = -1
        else: 
            k_b = k_s + 1        
                
        # Smaller values
        x_s = self.HIColumn[i_s]
        y_s = self.HeIColumn[j_s]
        z_s = self.HeIIColumn[k_s]
        
        # Bigger values
        x_b = self.HIColumn[i_b]
        y_b = self.HeIColumn[j_b]
        z_b = self.HeIIColumn[k_b]
                
        # Distance between supplied value and smallest value in table        
        x_d = (value[0] - x_s) / self.dHIColumn
        y_d = (value[1] - y_s) / self.dHeIColumn
        z_d = (value[2] - z_s) / self.dHeIIColumn
                
        return [i_s, j_s, k_s], [i_b, j_b, k_b], [x_d, y_d, z_d]
        
    def GetIndices4D(self, value, x_HII):
        """
        Return indices for 4D interpolation.
        """    
        
        x, y, z = self.GetIndices3D(value)
        
        if x_HII > self.esec.IonizedFractions[-1]:
            i = self.esec.NumberOfXiBins - 2
        elif x_HII <= self.esec.IonizedFractions[-1]:
            i = 0
        else:    
            i = 0
            while x_HII > self.esec.IonizedFractions[i]:
                i += 1
            
        return x, y, z, i   
        
