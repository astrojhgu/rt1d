"""

test_spectrum_fromfile.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sat Mar 23 16:07:21 2013

Description: Make sure reading spectrum from a file ('f') and passing spectrum
to rt1d directly ('d') as array both work.

"""

import pylab as pl
import numpy as np
import rt1d, h5py, os

sim = rt1d.run.RT(pf = {'problem_type': 2, 'stop_time': 10})
if not os.path.exists('bbspec.hdf5'):
    f = h5py.File('bbspec.hdf5')
    f.create_dataset('E', data=np.linspace(13.6, 100, 100))
    f.create_dataset('LE', data=np.array(map(sim.rs.Spectrum, 
        np.linspace(13.6, 100, 100))))
    f.close()

simf = rt1d.run.RT(pf = {'problem_type': 2, 'spectrum_file': 'bbspec.hdf5',
    'stop_time': 10})
simd = rt1d.run.RT(pf = {'problem_type': 2, 
    'spectrum_E': np.linspace(13.6, 100, 100),
    'spectrum_LE': np.array(map(sim.rs.Spectrum, 
        np.linspace(13.6, 100, 100))),
    'stop_time': 10})

anl = rt1d.analysis.Analyze(sim.checkpoints)
anlf = rt1d.analysis.Analyze(simf.checkpoints)
anld = rt1d.analysis.Analyze(simd.checkpoints)

ax = anl.TemperatureProfile(t=[1, 10])
anlf.TemperatureProfile(ax=ax, t=[1, 10], color='b', marker='o', s=25)
anld.TemperatureProfile(ax=ax, t=[1, 10], color='g', marker='s', s=150,
    facecolors='none')
pl.draw()        
raw_input('')


