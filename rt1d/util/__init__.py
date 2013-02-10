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

def sort(pf, prefix = 'spectrum', make_list = True):
    """
    Turn any item that starts with prefix_ into a list, if it isn't already.
    Hack off the prefix when we're done.
    """            
    
    result = {}
    for par in pf.keys():
        if par[0:len(prefix)] != prefix:
            continue
        
        new_name = par.partition('_')[-1]
        if type(pf[par]) is not list and make_list:
            result[new_name] = [pf[par]]
        else:
            result[new_name] = pf[par]
            
    return result       
    
#def uniquify(l):   
#    """
#    Return a revised version of 'list' containing only unique elements.  
#    This routine will preserve the order of the original list.
#    """
#    
#    def ID(x): 
#        return x 
#    
#    seen = {} 
#    result = [] 
#    for item in l: 
#        marker = ID(item) 
#        if marker in seen: 
#            continue 
#        
#        seen[marker] = 1 
#        result.append(item) 
#    
#    return np.array(result)
             