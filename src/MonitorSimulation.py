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
        self.MaxHIIFraction = pf["MaxHIIFraction"]
        self.MinHIFraction = 10**round(np.log10(1. - self.MaxHIIFraction), 2)
        self.StartRadius = pf["StartRadius"] * self.LengthUnits / cm_per_kpc
        
        # Construct r-array
        self.grid = np.arange(self.GridDimensions)
        self.r = self.grid * self.LengthUnits / self.GridDimensions / cm_per_kpc
    
    def Monitor(self, data, t):
        """
        Make some plots.
        """
                                        
        # A few fields
        x_H = data["HIDensity"] / (data["HIDensity"] + data["HIIDensity"])
        mi = min(min(x_H), 0.1 * self.MinHIFraction)
        T = data["Temperature"]
        
        # Clear previous figure
        pl.clf()
        
        # Plot neutral fraction and temperature
        pl.subplot(211)
        pl.loglog(self.r, x_H, color = 'k')
        pl.loglog([self.StartRadius, self.StartRadius], [mi, 1], linestyle = '--')
        pl.ylim(mi, 1)
        pl.ylabel(r'$x_H$')  
        pl.title("t = {0} Myr".format(t / s_per_myr))
        
        pl.subplot(212)
        pl.loglog(self.r, T, color = 'k')
        pl.loglog([self.StartRadius, self.StartRadius], [10, 1e6], linestyle = '--')
        pl.ylim(10, 1.5 * max(T))
        pl.xlabel(r'$r \ (\mathrm{kpc})$')
        pl.ylabel(r'$T \ (\mathrm{K})$')
        pl.draw()