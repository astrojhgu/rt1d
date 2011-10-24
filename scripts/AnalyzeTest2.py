"""
AnalyzeTest2.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2011-06-27.

Description: Plot radial profiles of neutral and ionized fractions, and temperature, for a few different times, as in 
RT06.

Notes: Supply parameter file as commmand line argument.  Assumes data dumps are 5 Myr apart. 
     
"""

import sys, misc
import rt1d.analysis as rta
from multiplot import *
from constants import *

ds = rta.Analyze(sys.argv[1])

mp = multiplot(dims = (2, 1), panel_size = (0.5, 1))

mp.axes[0].semilogy(ds.data[2].r / cm_per_kpc / 6.6, ds.data[2].x_HI, color = 'k', ls = '-', label = r'$1 - x_i$')
mp.axes[0].semilogy(ds.data[2].r / cm_per_kpc / 6.6, ds.data[2].x_HII, color = 'k', ls = '--', label = r'$x_i$')
mp.axes[0].semilogy(ds.data[20].r / cm_per_kpc / 6.6, ds.data[20].x_HI, color = 'k', ls = '-')
mp.axes[0].semilogy(ds.data[20].r / cm_per_kpc / 6.6, ds.data[20].x_HII, color = 'k', ls = '--')
mp.axes[0].semilogy(ds.data[100].r / cm_per_kpc / 6.6, ds.data[100].x_HI, color = 'k', ls = '-')
mp.axes[0].semilogy(ds.data[100].r / cm_per_kpc / 6.6, ds.data[100].x_HII, color = 'k', ls = '--')
mp.axes[1].semilogy(ds.data[2].r / cm_per_kpc / 6.6, ds.data[2].T, color = 'k', ls = '-')
mp.axes[1].semilogy(ds.data[20].r / cm_per_kpc / 6.6, ds.data[20].T, color = 'k', ls = '--')
mp.axes[1].semilogy(ds.data[100].r / cm_per_kpc / 6.6, ds.data[100].T, color = 'k', ls = ':')

mp.axes[0].set_xlim(0, 1.01)
mp.axes[1].set_xlim(0, 1.01)
mp.axes[1].set_ylim(3e3, 4e4)
mp.axes[1].set_xlabel(r'$r / L_{\mathrm{box}}$')
mp.axes[0].set_ylabel(r'$x_i$, $1-x_i$')
mp.axes[1].set_ylabel(r'$T \ (\mathrm{K})$')
mp.axes[0].legend(loc = 'lower right', frameon = False)

mp.fix_ticks()
mp.axes[0].set_ylim(1e-5, 1.5)
pl.savefig('{0}/RT_Test2_RadialProfiles.png'.format(ds.pf["OutputDirectory"]))
pl.savefig('{0}/RT_Test2_RadialProfiles.ps'.format(ds.pf["OutputDirectory"]))

misc.writetab((ds.data[2].r / cm_per_kpc / 6.6, ds.data[2].x_HI, ds.data[20].x_HI, ds.data[100].x_HI, \
    ds.data[2].T, ds.data[20].T, ds.data[100].T), \
    '{0}/RT_Test2_RadialProfiles.dat'.format(ds.pf["OutputDirectory"]), ('r/Lbox', 'x_HI (10 Myr)', 'x_HI (100 Myr)', 'x_HI (500 Myr)', \
    'T (10 Myr)', 'T (100 Myr)', 'T (500 Myr)'))
    
    
    
    
    