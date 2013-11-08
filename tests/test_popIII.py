"""

test_popIII.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Nov  7 17:47:29 MST 2013

Description: Test PopIII from birth, through death, and 10 Myr of its remnant
BHs lifetime.

"""

import rt1d
import pylab as pl

E_LyA = rt1d.physics.Constants.E_LyA
cm_per_kpc = rt1d.physics.Constants.cm_per_kpc

star_phase = \
{
 'problem_type': 2,
 'approx_lya': 0,
 'initial_redshift': 20.,
 'stop_time': 3.,
 'source_type': 'star',
 'source_qdot': 1e50,
 'source_temperature': 1e5,
 'expansion': 1,
 'grid_cells': 128,
 'start_radius': 0.01,
 'length_units': 25.*cm_per_kpc,
 'logarithmic_grid': False,
}

star_sim = rt1d.run.Simulation(**star_phase)
star_sim.run()
star_anl = rt1d.analyze.Simulation(star_sim.checkpoints)

ddf = star_sim.checkpoints.name(star_sim.checkpoints.stop_time)
zf = star_anl.data[ddf]['redshift']

bh_phase = \
{
 'problem_type': 2,
 'approx_lya': 0,
 'initial_redshift': zf,
 'stop_time': 20.,
 'source_type': 'bh',
 'source_mass': 100.,
 'source_rmax': 1e3,
 'source_evolving': 0,
 'spectrum_type': ['mcd', 'pl'],
 'spectrum_fraction': [0.5, 0.5],
 'spectrum_alpha': [None, -1.2],
 'spectrum_Emin': [E_LyA, 1e2],
 'spectrum_Emax': [1e4, 1e4],
 'spectrum_fcol': [1, None],
 'spectrum_logN': [0., 20.], 
 'expansion': 1,
 'grid_cells': 128,
 'start_radius': 0.01,
 'length_units': 25.*cm_per_kpc,
 'logarithmic_grid': False,
}

last_dd = star_sim.checkpoints.name(t=star_sim.checkpoints.DDtimes[-1])
last_ds = star_sim.checkpoints.data[last_dd]

bh_sim = rt1d.run.Simulation(ics=last_ds, grid=star_sim.grid, **bh_phase)
bh_sim.run()

bh_anl = rt1d.analyze.Simulation(bh_sim.checkpoints)

ax = star_anl.IonizationProfile(t=[1,2,3])
ax = bh_anl.IonizationProfile(t=[1,2,3,10], color='b')
raw_input('<enter> for temperature profiles')
pl.close()

ax = star_anl.TemperatureProfile(t=[1,2,3])
ax = bh_anl.TemperatureProfile(t=[1,2,3,10], color='b')
raw_input('<enter> for radial profiles of dTb')
pl.close()

ax = star_anl.BrightnessTemperatureProfile(t=[1,2,3])
ax = bh_anl.BrightnessTemperatureProfile(t=[1,2,3,10], color='b')
raw_input('<done>')

