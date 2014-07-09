"""

test_grid_logarithmic.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Nov  7 14:46:42 MST 2013

Description: Use a bright source to test logarithmic radial grid. Use 21-cm
profile as a gauge since there is small scale structure that a linear grid
may under-resolve.

"""

import rt1d
import matplotlib.pyplot as pl

E_LyA = rt1d.physics.Constants.E_LyA
cm_per_kpc = rt1d.physics.Constants.cm_per_kpc

pf = \
{
 'problem_type': 2,
 'approx_lya': 0,
 'initial_redshift': 20.,
 'stop_time': 10.,
 'source_type': 'bh',
 'source_mass': 1e4,
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
 'length_units': 100.*cm_per_kpc,
 
}

lin_sim = rt1d.run.Simulation(**pf)
lin_sim.run()

log_sim = rt1d.run.Simulation(logarithmic_grid=True, **pf)
log_sim.run()

lin_anl = rt1d.analyze.Simulation(lin_sim.checkpoints)
log_anl = rt1d.analyze.Simulation(log_sim.checkpoints)

ax = lin_anl.TemperatureProfile(t=[1,3,10], xscale='log')
ax = log_anl.TemperatureProfile(t=[1,3,10], xscale='log', ax=ax, color='b')
raw_input('<enter> for radial profiles of xHI & xHII')
pl.close()

ax = lin_anl.IonizationProfile(t=[1,3,10], xscale='log')
ax = log_anl.IonizationProfile(t=[1,3,10], xscale='log', ax=ax, color='b')
raw_input('<enter> for radial profiles of dTb')
pl.close()

ax = lin_anl.BrightnessTemperatureProfile(t=[1,3,10], xscale='log')
ax = log_anl.BrightnessTemperatureProfile(t=[1,3,10], xscale='log', ax=ax,
    color='b')
raw_input('<done>')

