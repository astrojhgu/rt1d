"""

AnalyzeSingleZone.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sun Apr 15 13:54:06 2012

Description: 

"""

import pylab as pl
import rt1d.analysis as rta

ds = rta.Analyze('./rt06_0.dat')

mp = rta.multiplot(dims = (2, 1), panel_size = (1, 1), useAxesGrid = False)

t, x_HI = ds.CellTimeEvolution(field = 'x_HI')
t, T = ds.CellTimeEvolution(field = 'T')

mp.grid[0].loglog(t / rta.s_per_yr, x_HI, color = 'k', ls = '-')
mp.grid[1].loglog(t / rta.s_per_yr, T, color = 'k', ls = '-')

for i in xrange(2):
    mp.grid[i].set_xlim(1e-6, 1e7)
    mp.grid[i].set_xscale('log')
    mp.grid[i].set_yscale('log')
    
mp.grid[0].set_ylim(1e-8, 1.5)
mp.grid[1].set_ylim(1e2, 1e5)
mp.fix_ticks()

mp.grid[0].set_ylabel(r'Neutral Fraction')
mp.grid[1].set_xlabel(r'$t / \mathrm{yr}$')
mp.grid[1].set_ylabel(r'$T \ (\mathrm{K})$')

pl.savefig('rt06_0.png')


