"""

CheckForParameterConflicts.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Dec 21 16:05:07 2011

Description: Check to make sure there are no conflicts between parameters.

"""

import numpy as np

known_conflicts = [(['InfiniteSpeedOfLight', 0], ['ParallelizationMethod', 1]),
                   (['TabulateIntegrals', 0], ['PhotonConserving', 0]),
                   (['DiscreteSpectrum', 0], ['PhotonConserving', 1], ['TabulateIntegrals', 0])]

def CheckForParameterConflicts(pf):
    """
    Loop over parameters and make sure none of them are in conflict.
    """
        
    probs = []
    for conflict in known_conflicts:
        
        ok = []
        for element in conflict:
            if pf[element[0]] == element[1]:
                ok.append(False)
            else:
                ok.append(True)
                                
        if not np.any(ok):
            probs.append(conflict)
        
    if probs:
        msg = []
        for con in probs:
            msg.append('%s = %g and %s = %g' % (con[0][0], con[0][1], con[1][0], con[1][1]))
        
        return True, msg
    else:
        return False, None      
            