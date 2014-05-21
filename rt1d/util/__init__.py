"""
Conversions between different naming conventions and other random stuff.
"""

import types
import numpy as np
from collections import Iterable
from .WriteData import CheckPoints
from .ProgressBar import ProgressBar
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
    
    if not kwargs:
        pf.update(ProblemType(1))
    elif 'problem_type' in kwargs:
        pf.update(ProblemType(kwargs['problem_type']))
    
    #for kwarg in kwargs:
    #    if kwarg not in defs.keys():
    #        print 'WARNING (rt1d): Unrecognized parameter: %s' % kwarg
    
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

def sort(pf, prefix='spectrum', make_list=True, make_array=False):
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
    if make_list or make_array:  
        lmax = 1
        for par in result:
            lmax = max(lmax, len(result[par]))
        
        for par in result:
            if len(result[par]) == lmax:
                continue
            
            result[par] = lmax * [result[par][0]]
            
            if make_array:
                result[par] = np.array(result[par])
            
    return result
    
class evolve:
    """ Make things that may or may not evolve with time callable. """
    def __init__(self, val):
        self.val = val
        self.callable = val == types.FunctionType
    def __call__(self, z = None):
        if self.callable:
            return self.val(z)
        else:
            return self.val
    
    
class ELEMENT:
    """ Substitute for periodic package, only knows about H and He. """
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
        
def readtab(fn, start = 0, stop = None, header = False, output_dict = True):
    """
    Returns columns from ASCII file as lists, or if header supplied (and
    output_dict = True) as a dictionary.
    """
    
    f = open(fn, 'r')
    
    data = []
    for i, line in enumerate(f):
        
        # Ignore blank
        if not line.strip(): 
            continue
        
        # Possibly read in header
        if line.split()[0][0] == '#': 
            if not header: 
                hdr = None            
            else: 
                hdr = line.split()[1:]
            
            continue
        
        # Only read between start and stop
        if (i + 1) < start: 
            continue
        if stop is not None:
            if (i + 1) > stop: 
                continue
                                    
        data.append(line.split())
        
        for j, element in enumerate(data[-1]):
            try: 
                data[-1][j] = float(element)
            except ValueError: 
                data[-1][j] = str(element)
    
    f.close()
    
    out = zip(*data)
    
    if header: 
        out.append(hdr)
        
        ret = {}
        for i, item in enumerate(hdr):
            ret[item] = out[i]
        
        if output_dict:    
            return ret    
            
    # Accommodate 1 column files
    if len(out) == 1 and not header: 
        return out[0]
    else: 
        return out
        
def ion_to_roman(ion):
    ints = (10, 9, 5, 4, 1)
    nums = ('X','IX','V','IV','I')
    result = ""
    for i in range(len(ints)):
       count = int(ion / ints[i])
       result += nums[i] * count
       ion -= ints[i] * count
    return result
    
def roman_to_ion(ion):
    nums = ['X', 'V', 'I']
    ints = [10, 5, 1]
    places = []
    for i in xrange(len(ion)):
       c = ion[i]
       value = ints[nums.index(c)]
       
       # If the next place holds a larger number, this value is negative.
       try:
          nextvalue = ints[nums.index(ion[i+1])]
          if nextvalue > value:
             value *= -1
       except IndexError:
          pass

       places.append(value)
       
    tot = 0
    for n in places: 
        tot += n
        
    # Easiest test for validity...
    if ion_to_roman(tot) == ion:
       return tot
    else:
       raise ValueError, 'input is not a valid roman numeral: %s' % input    

def convert_ion_name(name, convention='roman'):
    """ Convert species names to a particular convention."""

    if name in ['ge', 'de']:
        return name

    split = name.split('_')    
    if len(split) == 1:
        convention_in = 'roman'
    else:
        convention_in = 'underscore'
        
    if convention_in == convention:
        return name    
    
    if convention_in == 'underscore':
        element = split[0].upper()
        ion = ion_to_roman(int(split[1]))
        return '%s%s' % (element, ion)    

    # Isolate roman numerals
    element = name.split('I')[0]
    element = element.split('V')[0]
    element = element.split('X')[0]
    ion = name.split(element)[-1]
    
    return '%s_%s' % (element.lower(), roman_to_ion(ion))
            
def Gauss1D(x, pars):
    """ DOC """
    return pars[0] + pars[1] * np.exp(-(x - pars[2])**2 / 2. / pars[3]**2)

def boxcar(x, x1, x2):
    """ DOC """
    if hasattr(x, 'size'):
        y = 1.0 * np.ones_like(x)
        y[x < x1] = 0.0
        y[x > x2] = 0.0
        return y
        
    if x1 <= x <= x2:
        return 1.0
    
    return 0.0

def step(x, x0, pre_step=0.0):
    """ DOC """
    if hasattr(x, 'size'):
        y = (1.0 - pre_step) * np.ones_like(x)
        y[x <= x0] = pre_step
        return y
        
    if x <= x0:
        return 0.0
    
    return 1.0
    
    
                        