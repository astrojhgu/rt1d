"""

test_photon_conservation.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Jan 16 16:50:57 2013

Description: 

"""

import rt1d

m_H = rt1d.physics.constants.m_H
density_units = 1e-3 * m_H

# Control sims
pc = []

# Non-photon-conserving
npc = []

pcpf = {'problem_type': 2, 'grid_cells': 128}
npcpf = {'problem_type': 2, 'grid_cells': 128, 'photon_conserving': 0}

for i, x in enumerate([1, 2, 4, 8, 16]):
    sim1 = rt1d.run.RT(pf = pcpf.update({'density_units': density_units * x}))
    sim2 = rt1d.run.RT(pf = npcpf.update({'density_units': density_units * x})
    
    pc.append(rt1d.analysis.Analyze(sim1))
    pc2.append(rt1d.analysis.Analyze(sim12))
    
    del sim1, sim2
    
# Do some stuff    
    

