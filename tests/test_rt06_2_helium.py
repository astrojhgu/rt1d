"""

test_rt06_2_helium.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Dec 26 18:37:48 2012

Description: This is Test problem #2 from the Radiative Transfer
Comparison Project (Iliev et al. 2006; RT06) but with helium included.

"""

import rt1d, os
import matplotlib.pyplot as pl

pf = \
{
 'problem_type': 12,
 'approx_helium': 0,
 'source_table':'rt1d_integral_table.hdf5', 
 'tables_logNmin': [15]*3,
 'tables_logNmax': [20]*3,
 'tables_dlogN': [0.05]*3,
 'restricted_timestep': ['ions', 'neutrals', 'electrons', 'temperature'],
 'initial_timestep': 1e-6,
 "interp_method": 'linear',
 #'stop_time': 1e-5,
}

sim_He = rt1d.run.Simulation(**pf)
sim_He.run()

sim_H = rt1d.run.Simulation(problem_type=2)
sim_H.run()

anl_He = rt1d.analyze.Simulation(sim_He.checkpoints)
anl_H = rt1d.analyze.Simulation(sim_H.checkpoints)

fig1 = pl.figure(1); ax1 = fig1.add_subplot(111)
fig2 = pl.figure(2); ax2 = fig2.add_subplot(111)
fig3 = pl.figure(3); ax3 = fig3.add_subplot(111)

anl_H.TemperatureProfile(t=[10, 30, 100], ax=ax1, color='k', label='H-only')
anl_He.TemperatureProfile(t=[10, 30, 100], ax=ax1, color='b', label='H & He')


# Hydrogen profiles
anl_H.IonizationProfile(t=[10, 30, 100], ax=ax2, color='k', label='H-only')
anl_He.IonizationProfile(t=[10, 30, 100], ax=ax2, color='b', label='H & He')

# Helium profiles
anl_He.IonizationProfile(t=[10, 30, 100], annotate=True, species='he', ax=ax3, 
    color='k')

ax1.legend()
ax2.legend()
pl.draw()


