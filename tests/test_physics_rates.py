"""

test_physics_rates.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sun Apr 13 16:38:44 MDT 2014

Description: 

"""

import rt1d
import numpy as np
import pylab as pl

dims = 32
T = np.logspace(3, 5, 500)


colors = list('kb')
for i, src in enumerate(['fk96', 'chianti']):

    # Initialize grid object
    grid = rt1d.static.Grid(dims=dims)
    
    # Set initial conditions
    grid.set_physics(isothermal=True)
    grid.set_chemistry(Z=1)
    grid.set_density(rho0=rt1d.physics.Constants.m_H)
    grid.set_ionization()  
    grid.set_temperature(T)
    
    coeff = rt1d.physics.RateCoefficients(grid=grid, rate_src=src, T=T)
    
    CI = map(lambda TT: coeff.CollisionalIonizationRate(0, TT), T)
    RR = map(lambda TT: coeff.RadiativeRecombinationRate(0, TT), T)
    
    if i == 0:
        labels = [r'$\beta$', r'$\alpha$']
    else:
        labels = [None] * 2
    
    pl.loglog(T, CI, color=colors[i], ls='-', label=labels[0])
    pl.loglog(T, RR, color=colors[i], ls='--', label=labels[1])


pl.ylim(1e-18, 1e-8)
pl.legend(loc='upper left')  
pl.xlabel(r'Temperature $(\mathrm{K})$')
pl.ylabel(r'Rate $(\mathrm{cm}^{3} \ \mathrm{s}^{-1})$')
        
        