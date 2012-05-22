"""

CompareDiscretizationSchemes.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Tue May 22 09:25:38 2012

Description: Compare our optimal discrete SEDs to those constructed via
effective column density scheme of Aubert & Teyssier (2008). Plot modeled after
A&T Figure 4 (Test #2 from RT06).

"""

import numpy as np
import pylab as pl
import rt1d.analysis as rta

continuous = rta.Analyze('./continuous.dat')
optimal_mono = rta.Analyze('./optimal_monochromatic.dat')
effective_mono = rta.Analyze('./effective_xsection_monochromatic.dat')

Lbox = continuous.pf.LengthUnits

# Construct mask
mask = np.arange(0, len(continuous.data[0].r) - 1)
mask[mask % 2 != 0] = 0
mask[mask > 0] = 1
mask[1] = 1    
mask[-1] = 1  

mp = rta.multiplot(dims = (1, 2), useAxesGrid = False, panel_size = (2, 1), 
        share_all = False, aspect = False, padding = 0.2)

mp.grid[0].plot(continuous.data[1].r / Lbox, np.log10(continuous.data[1].T), color = 'k', ls = '-')
#mp.grid[0].plot(continuous.data[7].r / Lbox, np.log10(continuous.data[7].T), color = 'k', ls = '--')
mp.grid[0].plot(continuous.data[20].r / Lbox, np.log10(continuous.data[20].T), color = 'k', ls = ':')

mp.grid[0].scatter(optimal_mono.data[1].r[mask == 1] / Lbox, np.log10(optimal_mono.data[1].T[mask == 1]), color = 'b', marker = 'o', s = 20, facecolors = 'none')
#mp.grid[0].scatter(optimal_mono.data[7].r[mask == 1] / Lbox, np.log10(optimal_mono.data[7].T[mask == 1]), color = 'b', marker = '^', s = 20, facecolors = 'none')
mp.grid[0].scatter(optimal_mono.data[20].r[mask == 1] / Lbox, np.log10(optimal_mono.data[20].T[mask == 1]), color = 'b', marker = 's', s = 20, facecolors = 'none')

mp.grid[0].scatter(effective_mono.data[1].r[mask == 1] / Lbox, np.log10(effective_mono.data[1].T[mask == 1]), color = 'r', marker = 'o', s = 20, facecolors = 'none')
#mp.grid[0].scatter(effective_mono.data[7].r[mask == 1] / Lbox, np.log10(effective_mono.data[7].T[mask == 1]), color = 'r', marker = '^', s = 20, facecolors = 'none')
mp.grid[0].scatter(effective_mono.data[20].r[mask == 1] / Lbox, np.log10(effective_mono.data[20].T[mask == 1]), color = 'r', marker = 's', s = 20, facecolors = 'none')

mp.grid[0].set_xlim(0, 1)
mp.grid[0].set_ylim(3.5, 4.6)
mp.grid[0].set_xlabel(r'$r / L_{\mathrm{box}}$')
mp.grid[0].set_ylabel(r'$\mathrm{log}_{10}(T / \mathrm{K})$')

mp.grid[1].plot(continuous.data[1].r / Lbox, np.log10(continuous.data[1].x_HI), color = 'k', ls = '-')
#mp.grid[1].plot(continuous.data[7].r / Lbox, np.log10(continuous.data[7].x_HI), color = 'k', ls = '-')
mp.grid[1].plot(continuous.data[20].r / Lbox, np.log10(continuous.data[20].x_HI), color = 'k', ls = '-')
mp.grid[1].plot(continuous.data[1].r / Lbox, np.log10(continuous.data[1].x_HII), color = 'k', ls = '--')
#mp.grid[1].plot(continuous.data[7].r / Lbox, np.log10(continuous.data[7].x_HII), color = 'k', ls = '--')
mp.grid[1].plot(continuous.data[20].r / Lbox, np.log10(continuous.data[20].x_HII), color = 'k', ls = '--')

mp.grid[1].scatter(optimal_mono.data[1].r[mask == 1] / Lbox, np.log10(optimal_mono.data[1].x_HI[mask == 1]), color = 'b', marker = 'o', s = 20)
#mp.grid[1].scatter(optimal_mono.data[7].r[mask == 1] / Lbox, np.log10(optimal_mono.data[7].x_HI[mask == 1]), color = 'b', marker = '^', s = 20)
mp.grid[1].scatter(optimal_mono.data[20].r[mask == 1] / Lbox, np.log10(optimal_mono.data[20].x_HI[mask == 1]), color = 'b', marker = 's', s = 20)
mp.grid[1].scatter(optimal_mono.data[1].r[mask == 1] / Lbox, np.log10(optimal_mono.data[1].x_HII[mask == 1]), color = 'b', marker = 'o', s = 20, facecolors = 'none')
#mp.grid[1].scatter(optimal_mono.data[7].r[mask == 1] / Lbox, np.log10(optimal_mono.data[7].x_HII[mask == 1]), color = 'b', marker = '^', s = 20, facecolors = 'none')
mp.grid[1].scatter(optimal_mono.data[20].r[mask == 1] / Lbox, np.log10(optimal_mono.data[20].x_HII[mask == 1]), color = 'b', marker = 's', s = 20, facecolors = 'none')

mp.grid[1].scatter(effective_mono.data[1].r[mask == 1] / Lbox, np.log10(effective_mono.data[1].x_HI[mask == 1]), color = 'r', marker = 'o', s = 20)
#mp.grid[1].scatter(effective_mono.data[7].r[mask == 1] / Lbox, np.log10(effective_mono.data[7].x_HI[mask == 1]), color = 'r', marker = '^', s = 20)
mp.grid[1].scatter(effective_mono.data[20].r[mask == 1] / Lbox, np.log10(effective_mono.data[20].x_HI[mask == 1]), color = 'r', marker = 's', s = 20)
mp.grid[1].scatter(effective_mono.data[1].r[mask == 1] / Lbox, np.log10(effective_mono.data[1].x_HII[mask == 1]), color = 'r', marker = 'o', s = 20, facecolors = 'none')
#mp.grid[1].scatter(effective_mono.data[7].r[mask == 1] / Lbox, np.log10(effective_mono.data[7].x_HII[mask == 1]), color = 'r', marker = '^', s = 20, facecolors = 'none')
mp.grid[1].scatter(effective_mono.data[20].r[mask == 1] / Lbox, np.log10(effective_mono.data[20].x_HII[mask == 1]), color = 'r', marker = 's', s = 20, facecolors = 'none')

mp.grid[1].set_xlim(0, 1)
mp.grid[1].set_ylim(-5, 0.1)
mp.grid[1].set_xlabel(r'$r / L_{\mathrm{box}}$')
mp.grid[1].set_ylabel(r'$\mathrm{log}_{10}(x_i, 1 - x_i)$')

pl.draw()
pl.savefig('RT06_2_Neff_vs_sedop.png')
