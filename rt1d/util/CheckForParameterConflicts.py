"""

CheckForParameterConflicts.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Dec 21 16:05:07 2011

Description: Check to make sure there are no conflicts between parameters.

"""

import numpy as np

known_conflicts = [ \
    (['InfiniteSpeedOfLight', 0], ['ParallelizationMethod', 1]),
    (['Isothermal', 1], ['SecondaryIonization', 1])]

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
        for i, con in enumerate(probs):
            for element in con:
                msg.append('%s = %g' % (element[0], element[1]))
            
            if len(probs) > 1 and i != len(probs):
                msg.append('\nAND\n')
        
        return True, msg
    else:
        return False, None      
            