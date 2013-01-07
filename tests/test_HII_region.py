"""

test_HII_region.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Dec 26 18:37:48 2012

Description: This is Test problem #2 from the Radiative Transfer
Comparison Project (Iliev et al. 2006; RT06).

"""

import rt1d
import pylab as pl

sim = rt1d.run.RTsim(pf = {'problem_type': 2})

anl = rt1d.analysis.Analyze(sim.checkpoints)

anl.TemperatureProfile(t = [10, 100, 500])

raw_input('<enter> for radial profiles of xHI & xHII')

anl.IonizationProfile(t = [10, 100, 500], annotate = True)

raw_input('')

