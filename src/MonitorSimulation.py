"""
MonitorSimulation.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-11-19.

Description: Make plots of various quantities as a simulation runs so we can watch the progress more conveniently.
     
"""

import numpy as np
import pylab as pl

cm_per_kpc = 3.08568 * 10**21
s_per_myr = 365.25 * 24 * 3600 * 10**6

class MonitorSimulation:
    def __init__(self, pf):
        self.pf = pf
        self.GridDimensions = pf["GridDimensions"]
        self.LengthUnits = pf["LengthUnits"]
        self.TimeUnits = pf["TimeUnits"]
        self.InitialHIIFraction = pf["InitialHIIFraction"]
        
        self.grid = np.arange(self.GridDimensions)
        
        # Construct r-array
        self.r = self.grid * self.LengthUnits / self.GridDimensions / cm_per_kpc
    
    def Monitor(self, data, t):
        """
        Make some plots.
        """
                
        # Neutral fraction
        x_H = data["HIDensity"] / (data["HIDensity"] + data["HIIDensity"])
        
        pl.clf()
        pl.loglog(self.r, x_H, color = 'k')
        pl.xlabel(r'$r \ (\mathrm{kpc})$')
        pl.ylabel(r'$x_H$')  
        pl.title(r'$t = {0} \ \mathrm{Myr}$'.format(t))
        pl.draw()