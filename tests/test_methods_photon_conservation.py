"""

test_photon_conservation.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Jan 16 16:50:57 2013

Description: Run RT06 #2 with photon-conserving and non-photon-conserving
algorithms.

"""

import rt1d
import matplotlib.pyplot as pl

m_H = rt1d.physics.Constants.m_H
density_units = 1e-3 * m_H

# Control sims
pc = []

# Non-photon-conserving
npc = []

pcpf = {'problem_type': 2, 'grid_cells': 128}
npcpf = {'problem_type': 2, 'grid_cells': 128, 'photon_conserving': 0}

for i, x in enumerate([1, 2, 4, 8, 16]):
    units = density_units * x
    sim1 = rt1d.run.Simulation(pf = pcpf.update({'density_units': units}))
    sim2 = rt1d.run.Simulation(pf = npcpf.update({'density_units': units}))
    
    sim1.run()
    sim2.run()
    
    pc.append(rt1d.analyze.Simulation(sim1.checkpoints))
    npc.append(rt1d.analyze.Simulation(sim2.checkpoints))
    
    del sim1, sim2
    
# Do some stuff    
    

