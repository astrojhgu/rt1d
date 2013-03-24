"""
Conversions between different naming conventions and other random stuff.
"""


import numpy as np
from collections import Iterable
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
        if (isinstance(pf[par], Iterable) and type(pf[par]) is not str) \
            or (not make_list):
            result[new_name] = pf[par]
        elif make_list:
            result[new_name] = [pf[par]]
            
    # Make sure all elements are the same length?      
    if make_list:  
        lmax = 1
        for par in result:
            lmax = max(lmax, len(result[par]))
        
        for par in result:
            if len(result[par]) == lmax:
                continue
            
            result[par] = lmax * [result[par][0]]      
            
    return result
    
class ELEMENT:
    def __init__(self, name):
        self.name = name
    
    @property
    def mass(self):
        if not hasattr(self, '_mass'):
            if self.name == 'h':
                self._mass = 1.00794
            elif self.name == 'he':
                self._mass = 4.002602       
    
        return self._mass
    
class fake_chianti:
    def __init__(self):
        pass
    
    def z2element(self, i):
        if i == 1:
            return 'h'
        elif i == 2:
            return 'he'
        
    def element2z(self, name):
        if name == 'h':
            return 1
        elif name == 'he':
            return 2   
    
    def zion2name(self, Z, i):
        if Z == 1:
            if i == 1:
                return 'h_1'
            elif i == 2:
                return 'h_2'
        elif Z == 2:
            if i == 1:
                return 'he_1'
            elif i == 2:
                return 'he_2'
            elif i == 3:
                return 'he_3'             
    
    def convertName(self, species):
        element, i = species.split('_')
            
        tmp = {}
        tmp['Element'] = element
        tmp['Ion'] = self.zion2name(Z, int(i))
        tmp['Z'] = self.element2z(element)
        
        return tmp
                        