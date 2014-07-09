"""

test_background_radiation.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Tue Mar 26 22:10:28 2013

Description: Add constant background ionizing field.

"""

import rt1d
import numpy as np
import matplotlib.pyplot as pl

# RT06 #2
sim1 = rt1d.run.Simulation(problem_type=2, stop_time=30)
sim1.run()

# RT06 #2 + constant ionizing background
src1 = {'type': 'bb', 'Emin': 13.6, 'Emax': 1e2,
    'EminNorm': 0.0, 'EmaxNorm': np.inf}

pf = \
{
 'problem_type': 2,
 'stop_time': 30,
 'source_type': ['star', 'diffuse'],
 'source_ion': [None, 1e-15],
 'source_temperature': [1e5, None],
 'spectrum_pars': [src1, {'type': 6}],
}

sim2 = rt1d.run.Simulation(pf=pf)
sim2.run()

anl1 = rt1d.analyze.Simulation(sim1.checkpoints)
anl2 = rt1d.analyze.Simulation(sim2.checkpoints)

ax = anl1.IonizationProfile(t=[10, 30])
anl2.IonizationProfile(t=[10, 30], ax=ax, color='b')

