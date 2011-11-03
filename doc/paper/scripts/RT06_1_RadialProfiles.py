"""

RT06_1_RadialProfiles.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sun Sep 11 14:38:31 2011

Description: Plot neutral/ionized fraction vs. radius for test problem #1 at highest resolution

"""

import os
import pylab as pl
import rt1d.analysis as rta
from constants import *

RT1D = os.environ.get('RT1D')

ds = rta.Analyze("{0}/doc/examples/RT06_1_ResolutionTestSuite/c_infinite/dx6400_dt3.dat".format(RT1D))

ax = pl.subplot(111)
ax.semilogy(ds.data[0].r / cm_per_kpc / 6.6, ds.data[2].x_HII, color = 'k', ls = '--', label = r'$x_i$')
ax.semilogy(ds.data[0].r / cm_per_kpc / 6.6, ds.data[10].x_HII, color = 'k', ls = '--')
ax.semilogy(ds.data[0].r / cm_per_kpc / 6.6, ds.data[100].x_HII, color = 'k', ls = '--')
ax.semilogy(ds.data[0].r / cm_per_kpc / 6.6, ds.data[2].x_HI, color = 'k', ls = '-', label = r'$1 - x_i$')
ax.semilogy(ds.data[0].r / cm_per_kpc / 6.6, ds.data[10].x_HI, color = 'k', ls = '-')
ax.semilogy(ds.data[0].r / cm_per_kpc / 6.6, ds.data[100].x_HI, color = 'k', ls = '-')

ax.set_xlim(0, 1)
ax.set_ylim(1e-5, 1.2)

ax.set_xlabel(r'$r / L_{\mathrm{box}}$')
ax.set_ylabel(r'$x_i, 1 - x_i$')

pl.legend(loc = 'lower right', frameon = False)

# Annotate
r1 = - 0.02 + ds.LocateIonizationFront(2) / cm_per_kpc / 6.6
r2 = - 0.02 + ds.LocateIonizationFront(10) / cm_per_kpc / 6.6
r3 = - 0.02 + ds.LocateIonizationFront(100) / cm_per_kpc / 6.6

pl.annotate('10', (r1, 0.5), ha = 'right')
pl.annotate('100', (r2, 0.5), ha = 'right')
pl.annotate('500', (r3, 0.5), ha = 'right')

pl.draw()
pl.savefig('RT06_1_RadialProfiles.png')