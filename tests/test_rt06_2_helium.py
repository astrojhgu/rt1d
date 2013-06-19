"""

test_rt06_2.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Dec 26 18:37:48 2012

Description: This is Test problem #2 from the Radiative Transfer
Comparison Project (Iliev et al. 2006; RT06).

"""

import rt1d
import pylab as pl

sim = rt1d.run.Simulation(problem_type=12, approx_helium=1)
sim.run()

anl = rt1d.analyze.Simulation(sim.checkpoints)

ax = anl.TemperatureProfile(t=[10, 30, 100])
raw_input('<enter> for radial profiles of xHI & xHII')
pl.close()

ax = anl.IonizationProfile(t=[10, 30, 100])
raw_input('<enter> for radial profiles of xHI & xHII')

ax = anl.IonizationProfile(t=[10, 30, 100], species='he', color='b')
raw_input('')

