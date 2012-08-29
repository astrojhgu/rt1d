"""

AnalyzeTest1.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Aug 29 11:42:01 2012

Description: 

"""

import pylab as pl
import rt1d.analysis as rta

ds50 = rta.Analyze('./rt06_1_c50.dat')
ds100 = rta.Analyze('./rt06_1_c100.dat')
ds200 = rta.Analyze('./rt06_1_c200.dat')
ds400 = rta.Analyze('./rt06_1_c400.dat')

ds50.ComputeIonizationFrontEvolution()
ds100.ComputeIonizationFrontEvolution()
ds200.ComputeIonizationFrontEvolution()

# Plot it up
ds400.PlotIonizationFrontEvolution()

ds400.mp.grid[1].plot(ds50.t / ds50.trec, ds50.rIF / ds50.ranl, 
    color = 'r', ls = '--', label = r'$\Delta x = L_{\mathrm{box}} / 50')
ds400.mp.grid[1].plot(ds100.t / ds100.trec, ds100.rIF / ds100.ranl, 
    color = 'g', ls = '--', label = r'$\Delta x = L_{\mathrm{box}} / 100')
ds400.mp.grid[1].plot(ds200.t / ds200.trec, ds200.rIF / ds200.ranl, 
    color = 'b', ls = '--', label = r'$\Delta x = L_{\mathrm{box}} / 200')

pl.draw()
pl.savefig('rt06_1.png')
