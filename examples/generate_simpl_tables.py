"""

test_sed_mcd.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu May  2 10:46:44 2013

Description: Generate SIMPL accretion disk models. Save to HDF5 file.

"""

import rt1d, os, itertools
import numpy as np

try:
    from mpi4py import MPI
    rank, size = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size
except ImportError:
    rank, size = 0, 1

# 
## INPUT
rmax = 1e3
mass = [10., 100, 1e3]
f_scatter = [0.1, 0.5, 1.0]
gamma = [-2.5, -1.5, -0.5]
Emin = 10
Emax = 3e4
Nbins = 250
##
#

simpl = \
{
    'source_type': 'bh', 
    'source_mass': 10.,
    'source_rmax': rmax,
    'spectrum_type': 'simpl',
    'spectrum_Emin': Emin,
    'spectrum_Emax': Emax,
    'spectrum_alpha': -0.5,
    'spectrum_fsc': 1.0,
    'spectrum_logN': -np.inf,
}
    
for h, m in enumerate(mass):
    simpl.update({'source_mass': m})
    for i, fsc in enumerate(f_scatter):
        simpl.update({'spectrum_fsc': fsc})
        for j, alpha in enumerate(gamma):
            
            k = i * len(gamma) + j + 1
            
            if k % size != rank:
                continue
            
            simpl.update({'spectrum_alpha': alpha})
            
            bh_simpl = rt1d.sources.RadiationSource(init_tabs=False, **simpl)
            
            prefix = bh_simpl.sed_name()
                    
            E = np.logspace(np.log10(Emin), np.log10(Emax), Nbins)
            
            if os.path.exists('%s.txt' % prefix):
                print '%s.txt exists!' % prefix
                continue
                
            bh_simpl.dump('%s.txt' % prefix, E)
        