"""

CheckForParameterConflicts.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Dec 21 16:05:07 2011

Description: Check to make sure there are no conflicts between parameters.

"""


conflicts = [(['InfiniteSpeedOfLight', 0], ['ParallelizationMethod', 1]),
             (['DiscreteSpectrum', 0], ['PhotonConserving', 1])]

def CheckForParameterConflicts(pf):
    """
    Loop over parameters and make sure none of them are in conflict.
    """
    
    
    probs = []
    for con in conflicts:
        if (con[0][0] not in pf.keys()) and (con[1][0] not in pf.keys()): 
            continue
            
        if (pf[con[0][0]] == con[0][1]) and (pf[con[1][0]] == con[1][1]):
            probs.append(con)
        
    if probs:
        msg = []
        for con in probs:
            msg.append('%s = %g and %s = %g' % (con[0][0], con[0][1], con[1][0], con[1][1]))
        
        return True, msg
    else:
        return False, None      
            