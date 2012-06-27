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
    def __init__(self, pf, iits):
        self.pf = pf
        self.esec = SecondaryElectrons(pf)
        self.iits = iits
                
        self.locs = iits.locs
        self.dims = iits.dims        
        self.columns = iits.columns
        self.dcolumns = iits.dcolumns
        self.colmin = np.array(map(np.min, self.columns))
        self.colmax = np.array(map(np.max, self.columns))
        self.offsets = self.colmin / self.dcolumns
                
        # This is a dictionary with all the lookup tables
        self.itabs = iits.itabs
                
        # What kind of interpolator do we need? Where to look for values?
        if iits.Nd == 1:
            self.interp = self.InterpolateLinear
            self.GetIndices = self.GetIndices1D
        elif iits.Nd == 2:
            self.interp = self.InterpolateBiLinear
            self.GetIndices = self.GetIndices2D
        elif iits.Nd == 3:
            self.interp = self.InterpolateTriLinear
            self.GetIndices = self.GetIndices3D
        elif iits.Nd == 4:
            self.interp = self.InterpolateSphericalLinear
            self.GetIndices = self.GetIndices4D
        else:
            self.interp = self.InterpolateQuintLinear
            self.GetIndices = self.GetIndices5D    
                    
    # Do we still need this anywhere?        
    def OpticalDepth(self, value, species):
        return 10**np.interp(value, self.columns[species], self.itabs['logOpticalDepth%i' % species])
                
    def InterpolateLinear(self, indices, integral, value):
        """
        Use this technique for hydrogen-only calculations.  For consistency with MultiSpecies > 0 
        methods, 'value' should still be a 3-element list.  
        """    

        return 10**np.interp(value[0], self.columns[0], self.itabs[integral])
        
    def InterpolateBiLinear(self, indices, integral, value):
        """
        We use this for runs when MultiSpecies = 0 and (SecondaryIonization >= 2
        or SourceTimeEvolution > 0) with DiscreteSpectrum = 0 and TabulateIntegrals = 1.
        """    
        
        i_x, i_y = indices      
                                                                                        
        x1 = self.columns[0][i_x]
        x2 = self.columns[0][i_x + 1]
        y1 = self.columns[1][i_y]
        y2 = self.columns[1][i_y + 1]
                
        f11 = self.itabs[integral][i_x][i_y]
        f21 = self.itabs[integral][i_x + 1][i_y]
        f12 = self.itabs[integral][i_x][i_y + 1]
        f22 = self.itabs[integral][i_x + 1][i_y + 1]
                                        
        final = (f11 * (x2 - value[self.locs[0]]) * (y2 - value[self.locs[1]]) + \
            f21 * (value[self.locs[0]] - x1) * (y2 - value[self.locs[1]]) + \
            f12 * (x2 - value[self.locs[0]]) * (value[self.locs[1]] - y1) + \
            f22 * (value[self.locs[0]] - x1) * (value[self.locs[1]] - y1)) / \
            (x2 - x1) / (y2 - y1)
                                                                                         
        return 10**final    
        
    def InterpolateTriLinear(self, indices, integral, value):
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
    
    def InterpolateSphericalLinear(self, indices, integral, value):
        """
        This gets called if MultiSpecies = 1 and (SecondaryIonization >=2 or SourceTimeEvolution = 1)
        (and DiscreteSpectrum = 0, TabulateIntegrals = 1).
        """    
            
        i, j, k, l = indices
        
        results = np.zeros([2, 2, 2, 2])
        for ii in xrange(2):
            for jj in xrange(2):
                for kk in xrange(2):
                    for ll in xrange(2):
                        results[ii][jj][kk][ll] = self.itabs[integral][i + ii][j + jj][k + kk][l + ll]

        return np.mean(np.array(results))
                    
    def GetIndices1D(self, value = None):
        return None        
            
    def GetIndices2D(self, value):
        """
        Return column density and ionized fraction (or age) indices.
        """    
                
        i1 = int((value[self.locs[0]] - self.colmin[0]) / self.dcolumns[0])
        i2 = int((value[self.locs[1]] - self.colmin[1]) / self.dcolumns[1])
                
        return min(max(i1, 0), self.dims[0] - 2), min(max(i2, 0), self.dims[1] - 2)
        
    def GetIndices3D(self, value):
        """
        Retrieve set of 9 indices locating the interpolation points.        
        """
        
        if not self.pf['MultiSpecies']:
            return None
                                                                                                        
        # Smaller indices
        i_s = int((value[self.locs[0]] / self.dcolumns[0]) - self.offsets[0])
        j_s = int((value[self.locs[1]] / self.dcolumns[1]) - self.offsets[1])
        k_s = int((value[self.locs[2]] / self.dcolumns[2]) - self.offsets[2])
        
        # Bracketing coordinates - this might be slighty wrong. i_b = i_s + 1 always
        if i_s < 0: 
            i_s = i_b = 0
        elif i_s >= (self.dims[0] - 1): 
            i_s = i_b = -1
        else: 
            i_b = i_s + 1
        if j_s < 0: 
            j_s = j_b = 0
        elif j_s >= (self.dims[1] - 1): 
            j_s = j_b = -1
        else: 
            j_b = j_s + 1
        if k_s < 0: 
            k_s = k_b = 0
        elif k_s >= (self.dims[2] - 1): 
            k_s = k_b = -1
        else: 
            k_b = k_s + 1        
                
        # Smaller values
        x_s = self.columns[0][i_s]
        y_s = self.columns[1][j_s]
        z_s = self.columns[2][k_s]
        
        # Bigger values
        x_b = self.columns[0][i_b]
        y_b = self.columns[1][j_b]
        z_b = self.columns[2][k_b]
                
        # Distance between supplied value and smallest value in table        
        x_d = (value[self.locs[0]] - x_s) / self.dcolumns[0]
        y_d = (value[self.locs[1]] - y_s) / self.dcolumns[1]
        z_d = (value[self.locs[2]] - z_s) / self.dcolumns[2]
                
        return [i_s, j_s, k_s], [i_b, j_b, k_b], [x_d, y_d, z_d]
        
    def GetIndices4D(self, value):
        """
        Return 4 indices for 4D interpolation.
        """    
        
        i1 = int((value[self.locs[0]] - self.colmin[0]) / self.dcolumns[0])
        i2 = int((value[self.locs[1]] - self.colmin[1]) / self.dcolumns[1])
        i3 = int((value[self.locs[2]] - self.colmin[2]) / self.dcolumns[2])
        i4 = int((value[self.locs[3]] - self.colmin[3]) / self.dcolumns[3])        
                
        return min(max(i1, 0), self.dims[0] - 2), min(max(i2, 0), self.dims[1] - 2), \
            min(max(i3, 0), self.dims[2] - 2), min(max(i4, 0), self.dims[3] - 2)
                
    def GetIndices5D(self, value):
        i1 = int((value[self.locs[0]] - self.colmin[0]) / self.dcolumns[0])
        i2 = int((value[self.locs[1]] - self.colmin[1]) / self.dcolumns[1])
        i3 = int((value[self.locs[2]] - self.colmin[2]) / self.dcolumns[2])
        i4 = int((value[self.locs[3]] - self.colmin[3]) / self.dcolumns[3])        
        i5 = int((value[self.locs[4]] - self.colmin[4]) / self.dcolumns[4])        
                
        return min(max(i1, 0), self.dims[0] - 2), min(max(i2, 0), self.dims[1] - 2), \
            min(max(i3, 0), self.dims[2] - 2), min(max(i4, 0), self.dims[3] - 2), \
            min(max(i5, 0), self.dims[4] - 2)
        
