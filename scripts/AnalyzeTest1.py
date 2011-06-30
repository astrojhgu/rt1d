"""
AnalyzeTest1.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-11-09.

Description: Locate the position of the I-front (defined as x_H = 0.5) for each data dump in our 1D calculation, and
plot this position as a function of time (distance in kpc, time in Myr).  Also, plot neutral and ionized fractions
vs. radius and time.  Eventually compare with Enzo.

Notes: Supply parameter file as commmand line argument.     
     
"""

from multiplot import *
import os, re, h5py, sys
import numpy as np
import pylab as pl
import rt1d.analysis as rta

cm_per_kpc = 3.08568 * 10**21
s_per_myr = 365.25 * 24 * 3600 * 10**6

# First, read in conversion factors from parameter file
ds = rta.Dataset(sys.argv[1])
LengthUnits = ds.pf["LengthUnits"]
GridDims = ds.pf["GridDimensions"]
TimeUnits = ds.pf["TimeUnits"]
StartRadius = ds.pf["StartRadius"] * LengthUnits / cm_per_kpc
StartCell = ds.pf["StartRadius"] * GridDims
sigma_r = LengthUnits / GridDims / cm_per_kpc 

# Analytic solution
T = ds.data[0].T[0]
n_H = ds.data[0].n_HI[-1]
Ndot = ds.pf["SpectrumPhotonLuminosity"]
alpha_HII = 2.6e-13 * (T / 1.e4)**-0.85
trec = 1. / alpha_HII / n_H / s_per_myr
rs = (3. * Ndot / 4. / np.pi / alpha_HII / n_H**2)**(1. / 3.) / cm_per_kpc

# Array of cell indices
grid = np.arange(GridDims)

r = []
t = []
for dd in ds.data.keys():
    x_H = ds.data[dd].n_HI / (ds.data[dd].n_HI + ds.data[dd].n_HII)
    
    # Find position of I-front - interpolate over small range about min(x_H - 0.5)
    pd = abs(x_H - 0.5)
    pdpm = x_H - 0.5
    pos = list(pd).index(min(pd))
    tmp = [pdpm[pos - 1], pdpm[pos], pdpm[pos + 1]]
    
    newpos = np.interp(0.0, tmp, [pos - 1, pos, pos + 1])
    
    # Compute time (should be in code units)
    time = ds.data[dd].t / TimeUnits
                        
    r.append(newpos * LengthUnits / GridDims / cm_per_kpc)
    t.append(time * TimeUnits / s_per_myr)

r = np.array(r)    
t = np.array(t)

func = lambda t: rs * (1. - np.exp(-t / trec))**(1. / 3.) + StartRadius
t_anl = np.linspace(0, max(t), 500)
r_anl = map(func, t_anl)
r_anl_bin = map(func, t)
        
mp = multiplot(dims = (2, 1), panel_size = (0.5, 1))

mp.axes[0].plot(t / trec, r, ls = '--', color = 'k')
mp.axes[0].plot(t_anl / trec, r_anl, linestyle = '-', color = 'k')
mp.axes[0].set_xlim(0, 1 * max(t/trec))
mp.axes[0].set_ylim(0, 1.1 * max(max(r), max(r_anl)))
mp.axes[0].set_ylabel(r'$r \ (\mathrm{kpc})$')  

mp.axes[1].plot(t / trec, r / r_anl_bin, ls = '-', color = 'k')
mp.axes[1].set_xlim(0, 1 * max(t/trec))
mp.axes[1].set_ylim(0.9, 1.1)
mp.axes[1].set_xlabel(r'$t / t_{\mathrm{rec}}$')
mp.axes[1].set_ylabel(r'$r/r_{\mathrm{anl}}$') 
mp.fix_ticks()
pl.savefig('RT_Test1_IfrontEvolution.ps')
pl.savefig('RT_Test1_IfrontEvolution.png')
pl.clf()
        
# Ionized and neutral fractions vs. R and t (assumes dtDataDump = 5)
pl.semilogy(ds.data[0].r / cm_per_kpc / 6.6, ds.data[2].x_HI, ls = '-', color = 'k', label = r'$1 - x_i$')
pl.semilogy(ds.data[0].r / cm_per_kpc / 6.6, ds.data[6].x_HI, ls = '-', color = 'k')
pl.semilogy(ds.data[0].r / cm_per_kpc / 6.6, ds.data[20].x_HI, ls = '-', color = 'k')
pl.semilogy(ds.data[0].r / cm_per_kpc / 6.6, ds.data[100].x_HI, ls = '-', color = 'k')
pl.semilogy(ds.data[0].r / cm_per_kpc / 6.6, ds.data[2].x_HII, ls = '--', color = 'k', label = r'$x_i$')
pl.semilogy(ds.data[0].r / cm_per_kpc / 6.6, ds.data[6].x_HII, ls = '--', color = 'k')
pl.semilogy(ds.data[0].r / cm_per_kpc / 6.6, ds.data[20].x_HII, ls = '--', color = 'k')
pl.semilogy(ds.data[0].r / cm_per_kpc / 6.6, ds.data[100].x_HII, ls = '--', color = 'k')
pl.xlabel(r'$r / L_{\mathrm{box}}$')
pl.ylabel(r'$x_i$, $1-x_i$')
pl.xlim(0, 1.05)
pl.ylim(1e-5, 1.5)
pl.legend(loc = 'lower right', frameon = False)
pl.annotate('10', (0.29, 0.5))
pl.annotate('30', (0.45, 0.5))
pl.annotate('100', (0.655, 0.5))
pl.annotate('500', (0.78, 0.5))
pl.savefig('RT_Test1_RadialProfiles.ps')
pl.savefig('RT_Test1_RadialProfiles.png')
        