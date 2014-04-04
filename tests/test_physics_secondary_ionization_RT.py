"""

test_physics_secondary_ionization.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Dec 26 18:37:48 2012

Description: Compare results of RT06 #2 using 4 different methods of 
secondary ionization/heating.

"""

import rt1d
import pylab as pl

cm_per_kpc = rt1d.physics.Constants.cm_per_kpc

pf = {'problem_type': 2, 'grid_cells': 128, 'length_units': 10 * cm_per_kpc,
    'stop_time': 100, 'tables_dlogN': [0.1]}

fig1 = pl.figure(); ax1 = fig1.add_subplot(111)
fig2 = pl.figure(); ax2 = fig2.add_subplot(111)

colors = ['k', 'r', 'g', 'b']
for i in range(0, 4)[-1::-1]:
    pf.update({'secondary_ionization': i})
    sim = rt1d.run.Simulation(**pf)
    sim.run()

    anl = rt1d.analyze.Simulation(sim.checkpoints)

    anl.TemperatureProfile(t=[10, 100], color=colors[i], ax=ax1)
    anl.IonizationProfile(t=[10, 100], color=colors[i], ax=ax2)

    pl.draw()
    

