"""

test_rt06_2_helium.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Dec 26 18:37:48 2012

Description: This is Test problem #2 from the Radiative Transfer
Comparison Project (Iliev et al. 2006; RT06), only include helium.

"""

import rt1d
import pylab as pl

sim = rt1d.run.Simulation(pf = {'problem_type': 12})
sim.run()

anl = rt1d.analyze.Simulation(sim.checkpoints)

ax = anl.TemperatureProfile(t = [10, 100, 500])
raw_input('<enter> for radial profiles of xHI & xHII')
pl.close()

ax = anl.IonizationProfile(t = [10, 100, 500])
anl.IonizationProfile(ax=ax, t = [10, 100, 500], species='he', color='b')
raw_input('')

