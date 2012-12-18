"""

equil.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Sep 21 17:51:45 2012

Description: 

"""

import h5py, rt1d, os
import pylab as pl
import numpy as np

from mpi4py import MPI
rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

if not os.path.exists('oxygen_test'):
    os.mkdir('oxygen_test')
    
eq_dims = 100

# Read dataset
noneq = h5py.File('oxygen_high_timeres.h5', 'r')

grid = rt1d.Grid(dims = eq_dims)
grid.set_chem(Z = [8], abundance = 'cosmic', isothermal = True)
grid.set_rho(rho0 = rt1d.Constants.m_H)
grid.set_T(np.logspace(4, np.log10(5e6), eq_dims))
grid.set_x(state = 'equilibrium')

colors = ['k', 'b', 'g', 'r', 'm', 'c', 'y'] * 2

for i, key in enumerate(noneq.keys()):
    
    if i % size != rank:
        continue
    
    dd = int(key.strip('dd'))
    t = dd * 0.1   # kyr
    
    if t > 100:
        break
    
    ax = pl.subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e4, 5e6)
    ax.set_ylim(1e-8, 1.5)
    ax.set_title(r'$t = %.1f \ \mathrm{kyr}$' % t)
    ax.set_xlabel(r'$T \ (\mathrm{K})$')
    ax.set_ylabel(r'Oxygen Species Fraction')    

    for j, species in enumerate(grid.ion_species):
        ax.loglog(grid.data['T'], grid.data[species], color = colors[j])
        ax.scatter(noneq[key]['T'].value, noneq[key][species].value, 
            color = colors[j], marker = 'o', s = 50)           
            
    pl.savefig('oxygen_test/%s.png' % key)
    
    # Fun annotations
    if (t >= 10) and (t < 20):
        ax.annotate('come on guys!', xy = (2e4, 1e-3), ha = 'center', va = 'center',
            color = 'k')
    
    if (t >= 30) and (t < 40):
        ax.annotate('you can do it!', xy = (2e4, 1e-3), ha = 'center', va = 'center',
            color = 'k')
            
    if (t >= 50) and (t < 60):
        ax.annotate('that\'s it!', xy = (2e4, 1e-3), ha = 'center', va = 'center',
            color = 'k')        
    
    if (t >= 70) and (t < 80):
        ax.annotate('almost there!', xy = (2e4, 1e-3), ha = 'center', va = 'center',
            color = 'k')        
        
    if t == 100:
        ax.annotate('aww fuck it', xy = (2e4, 1e-3), ha = 'center', va = 'center',
            color = 'k')        
        
    pl.savefig('oxygen_test/%s_with_annotation.png' % key)     
    pl.close()
    del ax

