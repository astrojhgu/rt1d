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

import os, re, h5py, sys, misc
import numpy as np
import pylab as pl
import rt1d.analysis as rta

cm_per_kpc = 3.08568 * 10**21
s_per_myr = 365.25 * 24 * 3600 * 10**6

# Read in data
ds = rta.Analyze(sys.argv[1])
ds.ComputeIonizationFrontEvolution()

mp = rta.multiplot(dims = (2, 1), panel_size = (1, 1), useAxesGrid = False)

mp.grid[0].plot(ds.t / ds.trec, ds.rIF, color = 'k', ls = '--')
mp.grid[0].plot(ds.t / ds.trec, ds.ranl, linestyle = '-', color = 'k')
mp.grid[0].set_xlim(0, max(ds.t / ds.trec))
mp.grid[0].set_ylim(0, 1.1 * max(max(ds.rIF), max(ds.ranl)))
mp.grid[0].set_ylabel(r'$r \ (\mathrm{kpc})$')  

mp.grid[1].plot(ds.t / ds.trec, ds.rIF / ds.ranl, ls = '-', color = 'k')
mp.grid[1].set_xlim(0, max(ds.t / ds.trec))
mp.grid[1].set_ylim(0.9, 1.1)
mp.grid[1].set_xlabel(r'$t / t_{\mathrm{rec}}$')
mp.grid[1].set_ylabel(r'$r/r_{\mathrm{anl}}$') 

mp.grid[0].xaxis.set_ticks(np.linspace(0, 4, 5))
mp.grid[1].xaxis.set_ticks(np.linspace(0, 4, 5))

mp.fix_ticks()
pl.savefig('{0}/RT_Test1_IfrontEvolution.png'.format(ds.pf['OutputDirectory']))
pl.clf()

# Write out data
misc.writetab((ds.t / ds.trec, ds.r, ds.rIF / ds.ranl), '{0}/RT_Test1_IfrontEvolution.dat'.format(ds.pf['OutputDirectory']), ('t/trec', 'r', 'r/ranl'))

# Ionized and neutral fractions vs. R and t (assumes dtDataDump = 5)
pl.semilogy(ds.data[0].r / cm_per_kpc / 6.6, ds.data[2].x_HI, ls = '-', color = 'k', label = r'$1 - x_i$')
pl.semilogy(ds.data[0].r / cm_per_kpc / 6.6, ds.data[20].x_HI, ls = '-', color = 'k')
pl.semilogy(ds.data[0].r / cm_per_kpc / 6.6, ds.data[100].x_HI, ls = '-', color = 'k')
pl.semilogy(ds.data[0].r / cm_per_kpc / 6.6, ds.data[2].x_HII, ls = '--', color = 'k', label = r'$x_i$')
pl.semilogy(ds.data[0].r / cm_per_kpc / 6.6, ds.data[20].x_HII, ls = '--', color = 'k')
pl.semilogy(ds.data[0].r / cm_per_kpc / 6.6, ds.data[100].x_HII, ls = '--', color = 'k')

# Annotations for t = 10, 100, 500 Myr
a1 = ds.data[2].r[np.argmin(np.abs(ds.data[2].x_HI - 0.5))] / ds.pf['LengthUnits']
a2 = ds.data[20].r[np.argmin(np.abs(ds.data[20].x_HI - 0.5))] / ds.pf['LengthUnits']
a3 = ds.data[100].r[np.argmin(np.abs(ds.data[100].x_HI - 0.5))] / ds.pf['LengthUnits']

pl.xlabel(r'$r / L_{\mathrm{box}}$')
pl.ylabel(r'$x_i$, $1-x_i$')
pl.xlim(0, 1.01)
pl.ylim(1e-5, 1.5)
pl.legend(loc = 'lower right', frameon = False)
pl.annotate('10', (a1 - 0.05, 0.5))
pl.annotate('100', (a2 - 0.07, 0.5))
pl.annotate('500', (a3 - 0.07, 0.5))
pl.savefig('{0}/RT_Test1_RadialProfiles.png'.format(ds.pf['OutputDirectory']))
pl.clf()

# Write out data
misc.writetab((ds.data[0].r / cm_per_kpc / 6.6, ds.data[2].x_HI, ds.data[6].x_HI, ds.data[20].x_HI, ds.data[100].x_HI), 
    '{0}/RT_Test1_RadialProfiles.dat'.format(ds.pf['OutputDirectory']), ('r/Lbox', 'x_HI (10 Myr)', 'x_HI (30 Myr)', 'x_HI (100 Myr)', 'x_HI (500 Myr)'))
     