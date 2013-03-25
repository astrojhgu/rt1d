"""

test_secondary_ionization.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Dec 26 18:37:48 2012

Description: Compare results of RT06 #2 using 4 different methods of 
secondary ionization/heating.

"""

import rt1d
import pylab as pl

sim0 = rt1d.run.Simulation(pf = {'problem_type': 2})
sim0.run()

sim1 = rt1d.run.Simulation(pf = {'problem_type': 2, 'secondary_ionization': 1})
sim1.run()

sim2 = rt1d.run.Simulation(pf = {'problem_type': 2, 'secondary_ionization': 2})
sim2.run()

sim3 = rt1d.run.Simulation(pf = {'problem_type': 2, 'secondary_ionization': 3})
sim3.run()

anl0 = rt1d.analyze.Simulation(sim0.checkpoints)
anl1 = rt1d.analyze.Simulation(sim1.checkpoints)
anl2 = rt1d.analyze.Simulation(sim2.checkpoints)
anl3 = rt1d.analyze.Simulation(sim3.checkpoints)

ax = anl0.TemperatureProfile(t = [10, 100, 500])
anl1.TemperatureProfile(t = [10, 100, 500], color = 'b', ax = ax)
anl2.TemperatureProfile(t = [10, 100, 500], color = 'r', ax = ax)
anl3.TemperatureProfile(t = [10, 100, 500], color = 'g', ax = ax)

raw_input('<enter> for radial profiles of xHI & xHII')
pl.close()

ax = anl0.IonizationProfile(t = [10, 100, 500], annotate = True)
anl1.IonizationProfile(t = [10, 100, 500], color = 'b', ax = ax)
anl2.IonizationProfile(t = [10, 100, 500], color = 'r', ax = ax)
anl3.IonizationProfile(t = [10, 100, 500], color = 'g', ax = ax)

raw_input('')



