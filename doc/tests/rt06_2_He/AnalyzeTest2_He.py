"""

AnalyzeTest2_He.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Aug 29 11:53:44 2012

Description: Compare RT06 #2 solutions with and without including helium.

"""

import numpy as np
import pylab as pl
import rt1d.analysis as rta

dsH = rta.Analyze('./rt06_2_H.dat')
dsHe = rta.Analyze('./rt06_2_He.dat')

# 4-panel plot: ionization (top) and heating (bottom), 10 Myr (left), 100 Myr (right)
mp = rta.multiplot(dims = (2, 2), panel_size = (2, 2), useAxesGrid = False)

### UPPER PANELS
mp.grid[0].semilogy(dsH.data[0].r / rta.cm_per_kpc / 6.6, dsH.data[1].x_HII,   
    color = 'k', ls = '--')
mp.grid[0].semilogy(dsH.data[0].r / rta.cm_per_kpc / 6.6, dsH.data[1].x_HI,   
    color = 'k', ls = '-')        
mp.grid[0].semilogy(dsHe.data[0].r / rta.cm_per_kpc / 6.6, dsHe.data[1].x_HII,   
    color = 'b', ls = '--')
mp.grid[0].semilogy(dsHe.data[0].r / rta.cm_per_kpc / 6.6, dsHe.data[1].x_HI,   
    color = 'b', ls = '-')                          
mp.grid[1].semilogy(dsH.data[0].r / rta.cm_per_kpc / 6.6, dsH.data[10].x_HI,   
    color = 'k', ls = '-', label = r'$x_{\mathrm{HI}}$')   
mp.grid[1].semilogy(dsH.data[0].r / rta.cm_per_kpc / 6.6, dsH.data[10].x_HII,   
    color = 'k', ls = '--', label = r'$x_{\mathrm{HII}}$')
mp.grid[1].semilogy(dsHe.data[0].r / rta.cm_per_kpc / 6.6, dsHe.data[10].x_HI,   
    color = 'b', ls = '-')   
mp.grid[1].semilogy(dsHe.data[0].r / rta.cm_per_kpc / 6.6, dsHe.data[10].x_HII,   
    color = 'b', ls = '--')    
  
for i in xrange(2):
    mp.grid[i].set_ylim(1e-5, 1.5)
    mp.grid[i].set_xlim(0, 1)

### LOWER PANELS    
mp.grid[2].semilogy(dsH.data[0].r / rta.cm_per_kpc / 6.6, dsH.data[1].T,   
    color = 'k', ls = '-')          
mp.grid[2].semilogy(dsHe.data[0].r / rta.cm_per_kpc / 6.6, dsHe.data[1].T,   
    color = 'b', ls = '-')                                
mp.grid[3].semilogy(dsH.data[0].r / rta.cm_per_kpc / 6.6, dsH.data[10].T,
    color = 'k', ls = '-', label = 'H-only')
mp.grid[3].semilogy(dsHe.data[0].r / rta.cm_per_kpc / 6.6, dsHe.data[10].T,
    color = 'b', ls = '-', label = 'He included')    

for i in np.arange(2, 4):
    mp.grid[i].set_ylim(1e2, 4e4)    
    mp.grid[i].set_xlim(0, 1)
        
mp.fix_ticks()
mp.grid[1].set_yticklabels([])    

mp.grid[1].legend(loc = 'lower right', frameon = False, ncol = 1)
mp.grid[3].legend(loc = 'upper right', frameon = False, ncol = 1)
mp.grid[0].set_title(r'$t = 10 \ \mathrm{Myr}$', size = 'xx-large')
mp.grid[1].set_title(r'$t = 100 \ \mathrm{Myr}$', size = 'xx-large')
mp.grid[0].set_ylabel(r'Species Fraction', size = 'xx-large')
mp.grid[2].set_ylabel(r'$T \ (\mathrm{K})$', size = 'xx-large')
mp.global_xlabel(r'$r / L_{\mathrm{box}}$', xy = (0.5, 0.075), size = 'xx-large')
    
pl.savefig('rt06_2_He_effects.png')  
pl.close()

## Helium profiles
dsHe.IonizationProfile(t = [10, 30, 100], species = 'He', annotate = True)
dsHe.ax.set_ylim(1e-7, 1.5)
pl.draw()
pl.savefig('rt06_2_He_profiles.png')  
pl.close()
