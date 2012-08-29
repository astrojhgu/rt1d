"""

AnalyzeClumpTest.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Apr 11 14:13:11 2012

Description: 

"""

import pylab as pl
import rt1d.analysis as rta

mp = rta.multiplot(dims = (2, 1), useAxesGrid = False)
        
t = [1, 3, 5, 15]        
ls = ['-', ':', '--', '-.']

ds = rta.Analyze('rt06_3_c200.dat')

ct = 0
for dd in ds.data.keys():
    if ds.data[dd].t / ds.pf['TimeUnits'] not in t: 
        continue
        
    this_t = int(ds.data[dd].t / ds.pf['TimeUnits'])

    mp.grid[0].semilogy(ds.data[dd].r / ds.pf['LengthUnits'], ds.data[dd].x_HI, color = 'k', ls = ls[ct], 
        label = r'$t = %i \ \mathrm{Myr}$' % this_t)
    mp.grid[1].semilogy(ds.data[dd].r / ds.pf['LengthUnits'], ds.data[dd].T, color = 'k', ls = ls[ct])
    ct += 1
    
mp.grid[0].set_ylim(1e-3, 1.5)
mp.grid[1].set_ylim(10, 8e4)
                                        
for i in xrange(2):
    mp.grid[i].set_xlim(0.6, 1.0)
                            
mp.grid[1].set_xlabel(r'$x / L_{\mathrm{box}}$')    
mp.grid[0].set_ylabel('Neutral Fraction')
mp.grid[1].set_ylabel(r'Temperature $(K)$')    
mp.fix_ticks()

mp.grid[0].legend(loc = 'lower right', frameon = False)    
                
pl.draw() 

pl.savefig('rt06_3.png')