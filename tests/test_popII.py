"""

test_popII.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Nov  7 17:47:29 MST 2013

Description: 

"""

import rt1d
import matplotlib.pyplot as pl

E_LL = rt1d.physics.Constants.E_LL
E_LyA = rt1d.physics.Constants.E_LyA
cm_per_kpc = rt1d.physics.Constants.cm_per_kpc

star_phase = \
{
 'problem_type': 2,
 'approx_lya': 0,
 'initial_redshift': 20.,
 'stop_time': 10.,
 'source_type': 'star',
 'source_qdot': 1e50,
 'source_temperature': 1e4,
 'spectrum_Emin': E_LyA,
 'spectrum_Emax': 1e2, 
 'expansion': 1,
 'grid_cells': 128,
 'start_radius': 0.01,
 'length_units': 10.*cm_per_kpc,
 'logarithmic_grid': False,
}

sim = rt1d.run.Simulation(**star_phase)
sim.run()
anl = rt1d.analyze.Simulation(sim.checkpoints)

ax = anl.IonizationProfile(t=[1,5,10])
raw_input('<enter> for temperature profiles')
pl.close()

ax = anl.TemperatureProfile(t=[1,5,10])
raw_input('<enter> for radial profiles of dTb')
pl.close()

ax = anl.BrightnessTemperatureProfile(t=[1,5,10])
raw_input('<done>')

