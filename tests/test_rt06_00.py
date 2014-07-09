"""

test_rt06_00.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sun Dec 30 22:10:25 2012

Description: Test rate coefficients.

"""

import rt1d
import numpy as np
import matplotlib.pyplot as pl
import chianti.core as cc

pl.rcParams['legend.fontsize'] = 14

#
dims = 32
T = np.logspace(1, 7, dims)
#

# Chianti rates
h1 = cc.ion('h_1', T)
h2 = cc.ion('h_2', T)
he1 = cc.ion('he_1', T)
he2 = cc.ion('he_2', T)
he3 = cc.ion('he_3', T)

# Analytic fits
coeff = rt1d.physics.RateCoefficients()

"""
Plot collisional ionization rate coefficients.
"""

h1.diRate()
he1.diRate()
he2.diRate()

# Chianti
pl.loglog(T, h1.DiRate['rate'], color = 'k', ls = '-', 
    label = 'chianti')
pl.loglog(T, he1.DiRate['rate'], color = 'k', ls = '--')
pl.loglog(T, he2.DiRate['rate'], color = 'k', ls = ':')

# Fukugita '94
pl.loglog(T, coeff.CollisionalIonizationRate(0, T), color = 'b', ls = '-', 
    label = 'Fukugita & Kawasaki \'94')
pl.loglog(T, coeff.CollisionalIonizationRate(1, T), color = 'b', ls = '--')
pl.loglog(T, coeff.CollisionalIonizationRate(2, T), color = 'b', ls = ':')

pl.xlim(min(T), max(T))
pl.ylim(0.9 * min(h1.DiRate['rate']), 1.1 * max(h1.DiRate['rate']))
pl.xlabel(r'$T \ (\mathrm{K})$')
pl.ylabel(r'Collisional Ionization Rate $(\mathrm{cm}^3 \ \mathrm{s}^{-1})$')
pl.legend(loc = 'lower right', frameon = False)

raw_input('')
pl.close()

"""
Plot recombination rate coefficients.
"""

h2.rrRate()
he2.rrRate()
he2.drRate()
he3.rrRate()

# Chianti
pl.loglog(T, h2.RrRate['rate'], color = 'k', ls = '-', label = 'chianti')
pl.loglog(T, he2.RrRate['rate'], color = 'k', ls = '--')
pl.loglog(T, he3.RrRate['rate'], color = 'k', ls = ':')
pl.loglog(T, he2.DrRate['rate'], color = 'k', ls = '-.')

# Fukugita
pl.loglog(T, coeff.RadiativeRecombinationRate(0, T), color = 'b', ls = '-',
    label = 'Fukugita & Kawasaki \'94')
pl.loglog(T, coeff.RadiativeRecombinationRate(1, T), color = 'b', ls = '--')
pl.loglog(T, coeff.RadiativeRecombinationRate(2, T), color = 'b', ls = ':')
pl.loglog(T, coeff.DielectricRecombinationRate(T), color = 'b', ls = '-.')

pl.xlim(min(T), max(T))
pl.ylim(0.9 * min(h2.RrRate['rate']), 1.1 * max(coeff.DielectricRecombinationRate(T)))
pl.xlabel(r'$T \ (\mathrm{K})$')
pl.ylabel(r'Recombination Rate $(\mathrm{cm}^3 \ \mathrm{s}^{-1})$')

pl.legend(loc = 'lower center', frameon = False)

raw_input('')