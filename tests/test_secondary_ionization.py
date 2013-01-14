"""

test_HII_region.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Dec 26 18:37:48 2012

Description: 

"""

import rt1d
import pylab as pl

sim2 = rt1d.run.RTsim(pf = {'problem_type': 2, 'secondary_ionization': 2})
sim0 = rt1d.run.RTsim(pf = {'problem_type': 2})
sim1 = rt1d.run.RTsim(pf = {'problem_type': 2, 'secondary_ionization': 1,
    'source_table': sim0.rt.src.tabs})

anl0 = rt1d.analysis.Analyze(sim0.checkpoints)
anl1 = rt1d.analysis.Analyze(sim1.checkpoints)

ax = anl0.TemperatureProfile(t = [10, 100, 500])
anl1.TemperatureProfile(t = [10, 100, 500], color = 'b', ax = ax)

raw_input('<enter> for radial profiles of xHI & xHII')
pl.close()

ax = anl0.IonizationProfile(t = [10, 100, 500], annotate = True)
anl1.IonizationProfile(t = [10, 100, 500], color = 'b', ax = ax)

raw_input('')



