"""

AnalyzeSecondaryIonization.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Tue Apr 10 16:30:20 2012

Description: Analyze simulations in doc/tests/secondary_ionization. Run from
that directory.

"""

import pylab as pl
import rt1d.analysis as rta

si0 = rta.Analyze('prob2_SI0.dat')
si1 = rta.Analyze('prob2_SI1.dat')
si2 = rta.Analyze('prob2_SI2.dat')
si3 = rta.Analyze('prob2_SI3.dat')

mp = rta.multiplot(dims = (2, 2), panel_size = (2, 2), useAxesGrid = False)

mp.grid[0].semilogy(si0.data[0].r / rta.cm_per_kpc / 6.6, si0.data[1].x_HI, ls = '-', color = 'k')
mp.grid[0].semilogy(si1.data[0].r / rta.cm_per_kpc / 6.6, si1.data[1].x_HI, ls = '-', color = 'r')
mp.grid[0].semilogy(si2.data[0].r / rta.cm_per_kpc / 6.6, si2.data[1].x_HI, ls = '-', color = 'g')
mp.grid[0].semilogy(si3.data[0].r / rta.cm_per_kpc / 6.6, si3.data[1].x_HI, ls = '-', color = 'b')
mp.grid[0].semilogy(si0.data[0].r / rta.cm_per_kpc / 6.6, si0.data[1].x_HII, ls = '--', color = 'k')
mp.grid[0].semilogy(si1.data[0].r / rta.cm_per_kpc / 6.6, si1.data[1].x_HII, ls = '--', color = 'r')
mp.grid[0].semilogy(si2.data[0].r / rta.cm_per_kpc / 6.6, si2.data[1].x_HII, ls = '--', color = 'g')
mp.grid[0].semilogy(si3.data[0].r / rta.cm_per_kpc / 6.6, si3.data[1].x_HII, ls = '--', color = 'b')

mp.grid[1].semilogy(si0.data[0].r / rta.cm_per_kpc / 6.6, si0.data[10].x_HI, ls = '-', color = 'k', label = r'$x_{\mathrm{HI}}$')
mp.grid[1].semilogy(si1.data[0].r / rta.cm_per_kpc / 6.6, si1.data[10].x_HI, ls = '-', color = 'r')
mp.grid[1].semilogy(si2.data[0].r / rta.cm_per_kpc / 6.6, si2.data[10].x_HI, ls = '-', color = 'g')
mp.grid[1].semilogy(si3.data[0].r / rta.cm_per_kpc / 6.6, si3.data[10].x_HI, ls = '-', color = 'b')
mp.grid[1].semilogy(si0.data[0].r / rta.cm_per_kpc / 6.6, si0.data[10].x_HII, ls = '--', color = 'k', label = r'$x_{\mathrm{HII}}$')
mp.grid[1].semilogy(si1.data[0].r / rta.cm_per_kpc / 6.6, si1.data[10].x_HII, ls = '--', color = 'r')
mp.grid[1].semilogy(si2.data[0].r / rta.cm_per_kpc / 6.6, si2.data[10].x_HII, ls = '--', color = 'g')
mp.grid[1].semilogy(si3.data[0].r / rta.cm_per_kpc / 6.6, si3.data[10].x_HII, ls = '--', color = 'b')

mp.grid[2].semilogy(si0.data[0].r / rta.cm_per_kpc / 6.6, si0.data[1].T, ls = '-', color = 'k')
mp.grid[2].semilogy(si1.data[0].r / rta.cm_per_kpc / 6.6, si1.data[1].T, ls = '-', color = 'r')
mp.grid[2].semilogy(si2.data[0].r / rta.cm_per_kpc / 6.6, si2.data[1].T, ls = '-', color = 'g')
mp.grid[2].semilogy(si3.data[0].r / rta.cm_per_kpc / 6.6, si3.data[1].T, ls = '-', color = 'b')

mp.grid[3].semilogy(si0.data[0].r / rta.cm_per_kpc / 6.6, si0.data[10].T, ls = '-', color = 'k', 
    label = r'SecondaryIonization = 0')
mp.grid[3].semilogy(si1.data[0].r / rta.cm_per_kpc / 6.6, si1.data[10].T, ls = '-', color = 'r', 
    label = 'Shull & vanSteenberg (1985)')
mp.grid[3].semilogy(si2.data[0].r / rta.cm_per_kpc / 6.6, si2.data[10].T, ls = '-', color = 'g', 
    label = 'Ricotti et al. (2002)')
mp.grid[3].semilogy(si3.data[0].r / rta.cm_per_kpc / 6.6, si3.data[10].T, ls = '-', color = 'b', 
    label = 'Furlanetto & Stoever (2010)')

for i in xrange(2):
    mp.grid[i].set_ylim(1e-5, 1.5)
    mp.grid[i + 2].set_ylim(1e2, 5e4)
    
mp.grid[1].set_yticklabels([])    
mp.global_xlabel(r'$r / L_{\mathrm{box}}$')
mp.grid[0].set_ylabel(r'Species Fraction')
mp.grid[2].set_ylabel(r'$T (\mathrm{K})$')

mp.grid[0].set_title(r'$t = 10 \ \mathrm{Myr}$')
mp.grid[1].set_title(r'$t = 100 \ \mathrm{Myr}$')

mp.grid[1].legend(loc = 'lower right', frameon = False)
mp.grid[3].legend(loc = 'lower right', frameon = False)

mp.fix_ticks()

pl.draw()
pl.savefig('secondary_ionization_methods.png')
