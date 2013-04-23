"""

test_io_tables.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Tue Apr 16 20:31:55 2013

Description: Make sure we can write-out an integral table, read it in,
and successfully run a simultion with the version on disk.

"""

import os, rt1d

sim = rt1d.run.Simulation(problem_type=2)
sim.rs.all_sources[0].tab.dump('bbtab.hdf5')

sim2 = rt1d.run.Simulation(problem_type=2, source_table='bbtab.hdf5')
sim2.run()

# Don't need any more clutter in rt1d/tests
os.system('rm -f bbtab.hdf5')






