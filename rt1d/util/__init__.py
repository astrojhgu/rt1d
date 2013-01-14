"""
Conversions between different naming conventions and other random stuff.
"""

import numpy as np
from .WriteData import CheckPoints
from .ProblemTypes import ProblemType
from .SetDefaultParameterValues import *
from .ReadParameterFile import ReadParameterFile, ReadRestartFile
from .CheckForParameterConflicts import CheckForParameterConflicts

defs = SetAllDefaults()
def parse_kwargs(**kwargs):
    """
    Parse kwargs dictionary - populate with defaults.
    """    
    
    pf = defs.copy()
    
    if 'problem_type' in kwargs:
        pf.update(ProblemType(kwargs['problem_type']))
    
    pf.update(kwargs)
        
    conflicts = CheckForParameterConflicts(pf)

    if conflicts:
        raise Exception('Parameter combination not allowed.')

    return pf

def rebin(bins):
    """
    Take in an array of bin edges and convert them to bin centers.        
    """
    
    bins = np.array(bins)
    result = np.zeros(bins.size - 1)
    for i, element in enumerate(result):
        result[i] = (bins[i] + bins[i + 1]) / 2.
        
    return result
            