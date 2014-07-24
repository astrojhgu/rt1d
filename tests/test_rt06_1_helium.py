"""

test_rt06_1.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Dec 26 18:37:48 2012

Description: Not quite the same as RT06 #1, just the simplest check we can
do to see if helium RT works.

"""

import rt1d
import matplotlib.pyplot as pl

# HI, HeI, & HeII ionizing radiation
sim = rt1d.run.Simulation(problem_type=11, stop_time=50,
    spectrum_E=[54.4], spectrum_LE=[1.0])
sim.run()

fig1 = pl.figure(1); ax1 = fig1.add_subplot(111)
fig2 = pl.figure(2); ax2 = fig2.add_subplot(111)

anl = rt1d.analyze.Simulation(sim.checkpoints)

anl.IonizationProfile(t=[50], annotate=True, ax=ax1)
anl.IonizationProfile(t=[50], species='he', annotate=True,
    ax=ax2, color='k')

ax1.set_yscale('linear')
ax1.set_ylim(0, 1.05)    

ax2.set_yscale('linear')
ax2.set_ylim(0, 1.05)  

pl.draw()