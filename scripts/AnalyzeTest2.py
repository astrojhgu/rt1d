"""
AnalyzeTest2.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2011-06-27.

Description: Plot radial profiles of neutral and ionized fractions for a few different times, as in Wise & Abel (2011).

Notes: Supply parameter file as commmand line argument.     
     
"""

import sys
import rt1d.analysis as rta
from multiplot import *
from constants import *

t = [10., 30., 100., 500.]
mp = multiplot(dims = (2, 1), panel_size = (0.5, 1))
ds = rta.Dataset(sys.argv[1])
for dd in ds.data.keys():
    print ds.data[dd].t / ds.pf['TimeUnits']
    if ds.data[dd].t / ds.pf['TimeUnits'] not in t: continue
    
    mp.axes[0].semilogy(ds.data[dd].r / cm_per_kpc, ds.data[dd].x_HI, color = 'k', ls = '-')
    mp.axes[0].semilogy(ds.data[dd].r / cm_per_kpc, ds.data[dd].x_HII, color = 'k', ls = '--')
    mp.axes[1].semilogy(ds.data[dd].r / cm_per_kpc, ds.data[dd].T, color = 'k', ls = '-')

mp.axes[0].set_xlim(0, 6.6)
mp.axes[0].set_ylim(1e-5, 1)
mp.axes[0].set_ylabel(r'$x$')    
mp.axes[1].set_xlabel(r'$r$ (kpc)')
mp.axes[1].set_ylabel(r'$T$ (K)')
mp.axes[1].set_xlim(0, 6.6)
mp.axes[1].set_ylim(1e2, 1e5)
mp.fix_ticks()
pl.savefig('Test2.png')

    
    
    
    
    
    