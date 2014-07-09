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
 'approx_helium': 1,
 'source_table':'rt1d_integral_table.hdf5', 
 'tables_logNmin': [15]*3,
 'tables_logNmax': [20]*3,
 'tables_dlogN': [0.05]*3
}

sim = rt1d.run.Simulation(**pf)
sim.run()

anl = rt1d.analyze.Simulation(sim.checkpoints)

ax = anl.TemperatureProfile(t=[10, 30, 100])
raw_input('<enter> for radial profiles of xHI & xHII')
pl.close()

ax = anl.IonizationProfile(t=[10, 30, 100])
raw_input('<enter> for radial profiles of xHI & xHII')

ax = anl.IonizationProfile(t=[10, 30, 100], species='he', color='b')
raw_input('')

