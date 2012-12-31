"""
Conversions between different naming conventions and other random stuff.
"""

import numpy as np
from .WriteData import CheckPoints
from .SetDefaultParameterValues import *

def parse_kwargs(**kwargs):
    """
    Parse kwargs dictionary - populate with defaults.
    """    
    
    pf = SetAllDefaults()
    pf.update(kwargs)

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
            