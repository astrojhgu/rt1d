"""

test_io_spectrum.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sat Mar 23 16:07:21 2013

Description: Make sure reading spectrum from a file ('f') and passing spectrum
to rt1d directly ('d') as an array both work.

"""

import pylab as pl
import numpy as np
import rt1d, h5py, os

sim = rt1d.run.Simulation(pf = {'problem_type': 2, 'stop_time': 10})
if not os.path.exists('bbspec.hdf5'):
    f = h5py.File('bbspec.hdf5')
    f.create_dataset('E', data=np.linspace(13.6, 100, 100))
    f.create_dataset('LE', data=np.array(map(sim.rs.src[0].Spectrum, 
        np.linspace(13.6, 100, 100))))
    f.close()

# Read spectrum from file
simf = rt1d.run.Simulation(pf={'problem_type': 2, 'spectrum_file': 'bbspec.hdf5',
    'stop_time': 10})
    
# Read spectrum from array    
simd = rt1d.run.Simulation(pf={'problem_type': 2, 
    'spectrum_E': np.linspace(13.6, 100, 100),
    'spectrum_LE': np.array(map(sim.rs.src[0].Spectrum, 
        np.linspace(13.6, 100, 100))),
    'stop_time': 10})
        
# Run the sims to make sure we get the same answer (100 bins should be very close)        
simf.run()
simd.run()
sim.run()

anl = rt1d.analyze.Simulation(sim.checkpoints)
anlf = rt1d.analyze.Simulation(simf.checkpoints)
anld = rt1d.analyze.Simulation(simd.checkpoints)

ax = anl.TemperatureProfile(t=[1, 10])
anlf.TemperatureProfile(ax=ax, t=[1, 10], color='b', marker='o', s=25)
anld.TemperatureProfile(ax=ax, t=[1, 10], color='g', marker='s', s=150,
    facecolors='none')
pl.draw()        
raw_input('')

if os.path.exists('bbspec.hdf5'):
    os.remove('bbspec.hdf5')


