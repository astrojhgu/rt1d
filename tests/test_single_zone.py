"""

test_single_zone.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Jan 18 09:35:35 2013

Description: 

"""

import rt1d

sim = rt1d.run.RTsim(pf = {'problem_type': 0})

anl = rt1d.analysis.Analyze(sim.checkpoints)


