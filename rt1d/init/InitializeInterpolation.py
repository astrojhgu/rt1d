"""

InitializeInterpolation.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Jan 14 13:02:46 2013

Description: 

"""

import numpy as np
from scipy.interpolate import interp1d, interp2d, RectBivariateSpline, \
    LinearNDInterpolator

class LookupTable:
    def __init__(self, pf, name, logN, table, logx = None, t = None):
        self.name = name
        self.logN = np.array(logN)
        self.table = table
                
        if '_' in name:
            self.basename = name[0:name.find('_')]
        else:
            self.basename = name
        
        self.logNmin = np.min(self.logN, axis = 1)
        self.logNmax = np.max(self.logN, axis = 1)
        
        # Possible extra dimensions
        self.logx = logx
        self.t = t
                
        # Table properties
        self.shape = self.table.shape
        self.size = np.prod(self.shape)
        
        self.Nd = len(logN)
        self.evolving = int(np.any(pf['spectrum_evolving']))
        self.adv_secondary_ionization = int(pf['secondary_ionization'] > 1)
        self.Ed = self.adv_secondary_ionization + self.evolving
                
        self.D = self.Nd + self.Ed
                
        # Initialize
        self._init()
        
    def __call__(self, logN, logx = None, t = None):
        """
        Perform interpolation in self.D dimensions.
        """
        
        # If we are beyond bounds of integral table, fix    
        for i in xrange(self.Nd):
            logN[..., i][logN[..., i] < self.logNmin[i]] = self.logNmin[i]
            logN[..., i][logN[..., i] > self.logNmax[i]] = self.logNmax[i]
            
            
        if self.adv_secondary_ionization and logx is not None:
            logx[logx < self.logxmin] = self.logxmin
            logx[logx > self.logxmax] = self.logxmax
            
        # Compute result    
        if self.D == 1 or self.basename in ['logPhi', 'logTau']:
            logresult = self.interp(logN[...,0])
        elif self.D == 2:
            ax2 = self._extra_axis(logx, t)
            logresult = np.zeros_like(logN[..., 0])
            for i in xrange(logN.shape[0]):
                logresult[i] = self.interp(logN[i, 0], ax2[i]).squeeze()
            
        return logresult
            
    def _extra_axis(self, logx, t):
        if logx is not None:
            return logx
        
        return t        
        
    def _init(self):
        """
        Set up interpolation table.
        """
        
        # Set up interpolation tables
        if self.D == 1:
            self.interp = \
                interp1d(self.logN[0], self.table, kind = 'cubic')        
        elif self.D == 2:
            
            if self.adv_secondary_ionization:
                ax2 = self.logx
                self.logxmin = np.min(self.logx)
                self.logxmax = np.max(self.logx)
            elif self.evolving:
                ax2 = self.t
                                            
            if self.basename in ['logPhi', 'logTau'] and self.adv_secondary_ionization:
                self.interp = interp1d(self.logN[0], self.table)
            else:    
                self.interp = \
                    RectBivariateSpline(self.logN[0], ax2, self.table)
            
        else:    
            
            # or scipy.ndimage.map_coordinates
        
            Ngrid = []
            for element in self.tab.N:
                Ngrid.append(element * np.ones([len(element)] * self.tab.Nd))
                        
            Nrav = []
            for element in Ngrid:
                Nrav.append(element.ravel())
            
            pts = np.array(Nrav).T
                        
            #print pts.shape, self.tabs['logPhi_h_1'].ravel().shape
            
            self.tables = {}
            for tab in self.tabs:
                print tab, pts.shape, self.tabs[tab].ravel().shape
                self.tables[tab] = LinearNDInterpolator(pts, self.tabs[tab].ravel())
                    
        
        