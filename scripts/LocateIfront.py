"""
LocateIfront.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-11-09.

Description: Locate the position of the I-front (defined as x_H = 0.5) for each data dump in our 1D calculation, and
plot this position as a function of time (distance in kpc, time in Myr)
     
"""

import os, re, h5py
import numpy as np
import pylab as pl

cm_per_kpc = 3.08568 * 10**21
s_per_myr = 365.25 * 24 * 3600 * 10**6

# First, read in conversion factors from first data dump
f = h5py.File('dd0000.h5', 'r')
LengthUnits = f["ParameterFile"]["LengthUnits"].value
GridDims = f["ParameterFile"]["GridDimensions"].value
TimeUnits = f["ParameterFile"]["TimeUnits"].value
StartRadius = f["ParameterFile"]["StartRadius"].value * LengthUnits / cm_per_kpc
StartCell = f["ParameterFile"]["StartRadius"].value * GridDims
sigma_r = LengthUnits / GridDims / cm_per_kpc 

# Analytic solution
T = f["Data"]["Temperature"].value[0]
n_H = f["Data"]["HIDensity"].value[-1]
Ndot = f["ParameterFile"]["SpectrumPhotonLuminosity"].value
alpha_HII = 2.6e-13 * (T / 1.e4)**-0.85
trec = 1. / alpha_HII / n_H / s_per_myr
rs = (3. * Ndot / 4. / np.pi / alpha_HII / n_H**2)**(1. / 3.) / cm_per_kpc

# Array of cell indices
grid = np.arange(GridDims)

r = []
t = []
for dd in os.listdir(os.getcwd()):
    if not re.search('h5', dd): continue
    
    f = h5py.File(dd, 'r')
    x_H = f["Data"]["HIDensity"].value / (f["Data"]["HIDensity"].value + f["Data"]["HIIDensity"].value)
    
    # Find position of I-front - interpolate over small range about min(x_H - 0.5)
    pd = abs(x_H - 0.5)
    pdpm = x_H - 0.5
    pos = list(pd).index(min(pd))
    tmp = [pdpm[pos - 1], pdpm[pos], pdpm[pos + 1]]
    
    newpos = np.interp(0.0, tmp, [pos - 1, pos, pos + 1])
    
    # Read in time (should be in code units)
    time = f["ParameterFile"]["CurrentTime"].value
                        
    r.append(newpos * LengthUnits / GridDims / cm_per_kpc)
    t.append(time * TimeUnits / s_per_myr)
    f.close()

r = np.array(r)    
t = np.array(t)

func = lambda t: rs * (1. - np.exp(-t / trec))**(1. / 3.) + StartRadius
t_anl = np.linspace(0, max(t), 500)
r_anl = map(func, t_anl)
r_anl_bin = map(func, t)
    
error = 100. * np.divide(r - r_anl_bin, r_anl_bin)    
    
pl.subplot(211)    
pl.scatter(t / trec, r, marker = '+', color = 'blue', label = 'Numerical Solution')
pl.plot(t_anl / trec, r_anl, linestyle = '-', color = 'black', label = 'Analytic Solution')
pl.xlim(0, 1.1 * max(t/trec))
pl.ylim(0, 1.1 * max(max(r), max(r_anl)))
pl.ylabel(r'$r \ (\mathrm{kpc})$')  
#pl.legend(loc = 'lower right') 

pl.subplot(212)
pl.plot(t / trec, error)
pl.ylim(-10, 10)
pl.xlabel(r'$t / t_{rec}$')
pl.ylabel(r'$\% \ \mathrm{Error}$') 
pl.show()
