"""
rt1d.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-10-14.

Description: Driver script for a 1D radiative transfer code.

"""

import sys

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except:
    ImportError("Module mpi4py not found.  No worries, we'll just run in serial.")
    rank = 0
    size = 1

# Class definitions
from InitializeParameterSpace import *

# Retrieve name of parameter file from command line arguments.
pf = sys.argv[1]

# Instantiate initialization class, construct our N-D parameter space.
ips = InitializeParameterSpace(pf)
all_pfs = ips.AllParameterSets()
del pf

if rank == 0:
    print "**************************************************************"
    print "**                           rt1d                           **"
    print "**                                                          **"
    print "** Queue: {0} 1D radiative transfer calculation(s).           **".format(len(all_pfs)) 
    print "** Press ctrl-d to quit at any time.                        **" 
    print "**************************************************************"

# Loop over parameter sets. 
for i, pf in enumerate(all_pfs):
    if i % size != rank: continue
        
        
        
    

    


