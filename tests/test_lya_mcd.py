"""

test_lya_mcdpl.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sat Nov  2 14:42:26 MDT 2013

Description: 

"""

import rt1d
import pylab as pl

E_LyA = rt1d.physics.Constants.E_LyA
cm_per_kpc = rt1d.physics.Constants.cm_per_kpc

pf = \
{
 'problem_type': 2,
 'approx_lya': 0,
 'initial_redshift': 20.,
 'stop_time': 10,
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
 'grid_cells': 256,
 'start_radius': 0.01,
 'length_units': 100.*cm_per_kpc,
 
}

sim = rt1d.run.Simulation(**pf)
sim.run()

anl = rt1d.analyze.Simulation(sim.checkpoints)

ax = anl.TemperatureProfile(t=[1,3,10])
raw_input('<enter> for radial profiles of xHI & xHII')
pl.close()

ax = anl.IonizationProfile(t=[1,3,10])
raw_input('<enter> for radial profiles of dTb')
pl.close()

ax = anl.BrightnessTemperatureProfile(t=[1,3,10])
raw_input('<done>')