"""

AnalyzeResolutionSuite.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Sep  7 10:12:23 2011

Description: Compare results of resolution suites for RT1 and RT2.

Conventions: 
-Black solid line is reserved for the analytic I-front solution.
-Blue = highest grid resolution 
-Green = second highest
-Red = poorest
-Solid -> highest time resolution (adaptive)
-Dashed -> dt = 1e-2
-Dotted -> dt = 1e-1

*A dash-dotted yellow line will be assigned to something not fitting the above criteria

sys.argv[1] = ProblemType
sys.argv[2] = cinf or cfin

"""

import os, re, sys
import pylab as pl
import numpy as np
import rt1d.analysis as rta

pl.rcParams['figure.subplot.left'] = 0.12
pl.rcParams['figure.subplot.right'] = 0.88
pl.rcParams['legend.fontsize'] = 10

RT1D = os.environ.get('RT1D')
cinf = "{0}/doc/examples/RT06_{1}_ResolutionTestSuite/c_infinite".format(RT1D, int(sys.argv[1]))
cfin = "{0}/doc/examples/RT06_{1}_ResolutionTestSuite/c_finite".format(RT1D, int(sys.argv[1]))

if sys.argv[2] == 'cinf': path = cinf
else: path = cfin

pfs = ['dx100_dt3', 'dx200_dt3', 'dx400_dt3', 'dx800_dt3', 'dx1600_dt3', 'dx3200_dt3', 'dx6400_dt3']

# First, the c -> infinite tests
mp = rta.multiplot(dims = (2, 1), panel_size = (1, 1), useAxesGrid = False)
for pf in pfs:
    print "Loading {0}/{1}".format(path, pf)
    
    try: ds = rta.Analyze("{0}/{1}.dat".format(path, pf))
    except OSError: continue
    
    if not ds.pf["HIIRestrictedTimestep"]: continue
    if not os.path.exists("{0}/".format(cinf, ds.pf["OutputDirectory"])): continue

    if ds.pf['GridDimensions'] == 100: color = 'cyan'
    elif ds.pf['GridDimensions'] == 200: color = 'magenta'
    elif ds.pf['GridDimensions'] == 400: color = 'red'
    elif ds.pf['GridDimensions'] == 800: color = 'green'
    elif ds.pf['GridDimensions'] == 1600: color = 'blue'
    elif ds.pf['GridDimensions'] == 3200: color = 'gray'
    else: color = 'yellow' 
                                                      
    ds.ComputeIonizationFrontEvolution()
        
    if ds.pf['GridDimensions'] == 6400: 
        mp.grid[0].plot(ds.t / ds.trec, ds.ranl, linestyle = '-', color = 'k', label = r'$r_{\mathrm{anl}}$')
        mp.grid[0].plot(ds.t / ds.trec, ds.rIF, color = 'k', ls = '--', label = r'$r_{\mathrm{num}}$')
        mp.grid[1].plot(ds.t / ds.trec, ds.rIF / ds.ranl, color = 'k', ls = '-', label = r'$\Delta x = 1/6400$')
        
    else:
        mp.grid[1].plot(ds.t / ds.trec, ds.rIF / ds.ranl, color = color, ls = '-', label = r'$\Delta x = 1/{0}$'.format(int(ds.pf['GridDimensions'])))

mp.grid[1].set_ylim(0.9, 1.05) 
mp.fix_ticks()     
mp.grid[0].set_xlim(0, max(ds.t / ds.trec))
mp.grid[1].set_xlim(0, max(ds.t / ds.trec))
mp.grid[1].yaxis.set_ticks(np.linspace(0.91, 1.05, 8), minor = True)    
mp.grid[0].set_ylabel(r'$r \ (\mathrm{kpc})$') 
mp.grid[1].set_xlabel(r'$t / t_{\mathrm{rec}}$')
mp.grid[1].set_ylabel(r'$r_{\mathrm{num}} /r_{\mathrm{anl}}$')
mp.grid[1].legend(loc = 'lower right', frameon = False, ncol = 2)

pl.rcParams['legend.numpoints'] = 10
pl.rcParams['legend.fontsize'] = 12
mp.grid[0].legend(loc = 'lower right', frameon = False, ncol = 2)

pl.draw()
pl.savefig('RT06_{0}_IfrontEvolution.png'.format(int(sys.argv[1])))


