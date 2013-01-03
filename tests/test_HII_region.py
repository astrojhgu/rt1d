"""

test_HII_region.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Dec 26 18:37:48 2012

Description: 

"""

import rt1d
import pylab as pl

sim = rt1d.run.RTsim(pf = {'problem_type': 2})
anl = rt1d.analysis.Analyze(sim.checkpoints)
anl.TemperatureProfile(t = [10, 30, 100])

raw_input('<enter> for radial profiles of xHI & xHII')
pl.close()
del anl.ax

anl.IonizationProfile(t = [10, 30, 100], annotate = True)

raw_input('')

