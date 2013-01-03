"""

test_HII_region_isoT.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Dec 26 18:37:48 2012

Description: 

"""

import rt1d
import pylab as pl

sim = rt1d.run.RTsim(pf = {'problem_type': 1})
anl = rt1d.analysis.Analyze(sim.checkpoints)
anl.PlotIonizationFrontEvolution()

raw_input('<enter> for radial profiles of xHI & xHII')
pl.close()
del anl.mp

anl.IonizationProfile(t = [10, 30, 100], annotate = True)

raw_input('')
