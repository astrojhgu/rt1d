"""

test_physics_secondary_ionization_coeff.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Apr  3 16:58:43 MDT 2014

Description: Reproduce Figures 2-3 in Furlanetto & Stoever (2010).

"""

import rt1d
import pylab as pl
import numpy as np
from multiplot import multipanel

# First, compare at fixed ionized fraction
xe = [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 0.9]
E = np.logspace(1, 4, 400)
channels = ['heat', 'h_1', 'lya', 'exc']
channels.reverse()

colors = ['k', 'b', 'r', 'g', 'm', 'c', 'y']

mp = multipanel(dims=(2, 2), padding=(0.2, 0.2))

esec = rt1d.physics.SecondaryElectrons(method=3)

for j, channel in enumerate(channels):
    
    for k, x in enumerate(xe):       
        
        if j == 2:
            if x < 0.5:
                label = r'$x_e = 10^{%i}$' % (np.log10(x))
            else:
                label = r'$x_e = %.2g$' % x
        else:
            label = None                       
             
        f = map(lambda EE: esec.DepositionFraction(xHII=x, E=EE, 
            channel=channel), E)
        
        mp.grid[j].semilogx(E, f, color=colors[k], ls='-', label=label)
        
    mp.grid[j].set_ylabel(r'$f_{\mathrm{%s}}$' % channel)
    mp.grid[j].set_yscale('linear')
    mp.grid[j].set_ylim(0, 1.05)
    
    if j == 2:
        mp.grid[2].legend(loc='upper left')

for i in range(3):
    mp.grid[i].set_xlabel(r'Electron Energy (eV)')

pl.draw()