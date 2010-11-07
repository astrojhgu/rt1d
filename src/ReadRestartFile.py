"""
ReadRestartFile.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-11-06.

Description: 

Notes: 
     
"""

import h5py

def ReadRestartFile(rf):
    f = h5py.File(rf, 'r')
    
    pf = {}
    data = {}
    for parameter in f["ParameterFile"]:
        pf[parameter] = f["ParameterFile"][parameter].value
        
    for field in f["Data"]:
        data[field] = f["Data"][field].value
        
    return pf, data
        
    