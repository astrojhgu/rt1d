"""
ReadParameterFile.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-10-14.

Description: Read rt1d parameter file, convert to python dictionary.

Note on problem types: 
Integer problem types are reserved for continuous spectra, while non-integer
problem types will refer to the same problem with DiscreteSpectrum = 1.  The
only exception to this rule is ProblemType = 1 (I-front propagation around
monochromatic source of photons).
     
"""

import re, copy
import numpy as np
from .ProblemTypes import ProblemType
from .SetDefaultParameterValues import SetAllDefaults

try:
    import h5py
except ImportError:
    pass

def ReadParameterFile(pf, setdefs = True):
    """
    Read in the parameter file, and parse the parameter names and arguments.
    Return a dictionary that contains all parameters and their values, whether 
    they be floats, tuples, or lists.
    """
    
    # Set defaults (possibly)
    if setdefs:
        pf_dict = SetAllDefaults()
    else:
        pf_dict = {}
    
    # Read file 
    f = open(pf, "r")   
    for line in f:
        if not line.split(): 
            continue
        if line.split()[0][0] == "#": 
            continue
        
        # This will prevent crashes if there is not a blank line at the end of the parameter file
        if line[-1] != '\n': 
            line += '\n'
        
        # Cleave off end-of-line comments.
        line = line[:line.rfind("#")].strip()
        
        # Read in the parameter name and the parameter value(s).
        parname, eq, parval = line.partition("=")
                        
        # ProblemType option
        if parname.strip() == 'ProblemType':
            pf_new = ProblemType(float(parval))
            for param in pf_new: 
                pf_dict[param] = pf_new[param]
                        
        # Else, actually read in the parameter           
        try: 
            parval = float(parval)
        except ValueError:
            if parval.replace('_', '').replace('.', '').strip().isalnum(): 
                parval = str(parval.strip())
            elif re.search('/', parval):
                parval = str(parval.strip())
            elif not parval.replace('.', '').strip():
                parval = str(parval).strip()    
            else:
                parval = parval.strip().split(",")
                tmp = []                           
                if parval[0][0] == '[':
                    for element in parval: 
                        new_element = element.strip("[,]")
                        try:
                            tmp.append(float(new_element.strip()))
                        except ValueError:
                            tmp.append(str(new_element.strip()))
                    parval = list(tmp)
                else:
                    raise ValueError('The format of this parameter is not understood.')
        
        pf_dict[parname.strip()] = parval
                    
    return pf_dict
    
def ReadRestartFile(rf):
    
    # First, the parameter file
    pf = ReadParameterFile(rf)
    
    # Now, the data
    f = h5py.File('%s.h5' % rf, 'r')
    
    data = {} 
    for field in f["data"]:
        data[field] = f["data"][field].value
            
    return pf, data       


    