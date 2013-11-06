"""

test_lya.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sat Nov  2 14:42:26 MDT 2013

Description: 

"""

import rt1d
import pylab as pl

E_LL = rt1d.physics.Constants.E_LL
E_LyA = rt1d.physics.Constants.E_LyA
cm_per_kpc = rt1d.physics.Constants.cm_per_kpc

sim = rt1d.run.Simulation(problem_type=2, approx_lya=0, spectrum_Emin=E_LyA,
    expansion=1, initial_redshift=20., stop_time=3., 
    length_units=10.*cm_per_kpc, grid_cells=128)
sim.run()

anl = rt1d.analyze.Simulation(sim.checkpoints)

ax = anl.TemperatureProfile(t=[1,3])
raw_input('<enter> for radial profiles of xHI & xHII')
pl.close()

ax = anl.IonizationProfile(t=[1,3])
raw_input('<enter> for radial profiles of dTb')
pl.close()

ax = anl.BrightnessTemperatureProfile(t=[1,3])
raw_input('<done>')