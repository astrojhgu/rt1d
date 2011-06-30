"""
ReadParameterFile.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-10-14.

Description: 
     
"""

import copy, h5py
import numpy as np
from rt1d.mods.SetDefaultParameterValues import SetDefaultParameterValues

cm_per_kpc = 3.08568 * 10**21
s_per_myr = 365.25 * 24 * 3600 * 10**6

def ReadParameterFile(pf):
    """
    Read in the parameter file, and parse the parameter names and arguments.
    Return a dictionary that contains all parameters and their values, whether 
    they be floats, tuples, or lists.
    """
    f = open(pf, "r")
    pf_dict = SetDefaultParameterValues()
    for line in f:
        if not line.split(): continue
        if line.split()[0][0] == "#": continue
        
        # Cleave off end-of-line comments.
        line = line[:line.rfind("#")].strip()
        
        # Read in the parameter name and the parameter value(s).
        parname, eq, parval = line.partition("=")
                    
        # ProblemType option
        if parname.strip() == 'ProblemType' and float(parval) > 0:
            pf_new = ProblemType(float(parval))
            for param in pf_new: pf_dict[param] = pf_new[param]
            
        # Else, actually read in the parameter                                     
        try: parval = float(parval)
        except ValueError:
            if parval.strip().isalnum(): 
                parval = str(parval.strip())
            else:
                parval = parval.strip().split(",")
                tmp = []                           
                if parval[0][0] == '[':
                    for element in parval: tmp.append(float(element.strip("[,]")))
                    parval = list(tmp)
                else:
                    raise ValueError('The format of this parameter is not understood.')
                
        pf_dict[parname.strip()] = parval
                
    return pf_dict
    
def ReadRestartFile(rf):
    f = h5py.File(rf, 'r')
    
    pf = {}
    data = {}
    for parameter in f["ParameterFile"]:
        pf[parameter] = f["ParameterFile"][parameter].value
        
    for field in f["Data"]:
        data[field] = f["Data"][field].value
        
    return pf, data    
    
def ProblemType(pt):
    """
    Storage bin for predefined problem types, 'pt's, like those used in the radiative transfer comparison project ('RT'),
    or John and Tom's 2010 ENZO-MORAY ('EM') paper.
    """
    
    # RT06-1, RT1: Pure hydrogen, isothermal HII region expansion
    if pt == 1.0:
        pf = {"ProblemType": 1, "InterpolationMethod": 0, \
              "ColumnDensityBinsHI": 500, "GridDimensions": 100, "LengthUnits": 6.6 * cm_per_kpc, \
              "TimeUnits": s_per_myr, "CurrentTime": 0.0, "StopTime": 500.0, \
              "StartRadius": 0.01, "dtDataDump": 5.0, "DataDumpName": 'dd', \
              "SavePrefix": 'rt', "Isothermal": 1, "MultiSpecies": 0, "SecondaryIonization": 0, "CosmologicalExpansion": 0, \
              "DensityProfile": 0, "InitialDensity": 1e-3, "TemperatureProfile": 0, "InitialTemperature": 1e4, \
              "IonizationProfile": 1, "InitialHIIFraction": 1.2e-3, "SourceType": 0, "SourceLifetime": 1e10, \
              "SpectrumPhotonLuminosity": 5e48, "DiscreteSpectrumMethod": 1, "DiscreteSpectrumSED": [13.6], \
              "SpectrumMinEnergy": 0.1, "SpectrumMaxEnergy": 100, "CollisionalIonization": 0
             }        

    # RT06-2: Pure hydrogen, HII region expansion, temperature evolution allowed, continuous BB spectrum
    if pt == 2.0:
       pf = {"ProblemType": 2, "InterpolationMethod": 0, \
             "ColumnDensityBinsHI": 500, "GridDimensions": 100, "LengthUnits": 6.6 * cm_per_kpc, \
             "TimeUnits": s_per_myr, "CurrentTime": 0.0, "StopTime": 500.0, \
             "StartRadius": 0.01, "dtDataDump": 5.0, "DataDumpName": 'dd', \
             "Isothermal": 0, "MultiSpecies": 0, "SecondaryIonization": 0, "CosmologicalExpansion": 0, \
             "DensityProfile": 0, "InitialDensity": 1e-3, "TemperatureProfile": 0, "InitialTemperature": 1e2, \
             "IonizationProfile": 1, "InitialHIIFraction": 0, "SourceType": 0, "SourceLifetime": 1e10, \
             "SpectrumPhotonLuminosity": 5e48, "DiscreteSpectrumMethod": 0, \
             "DiscreteSpectrumSED": [16.74, 24.65, 34.49, 52.06], "DiscreteSpectrumRelLum": [0.277, 0.335, 0.2, 0.188]
            }                  
    
    # EM-3: I-front trapping in a dense clump
    if pt == 3:
        pf = {"ProblemType": 3, "InterpolationMethod": 0, \
              "ColumnDensityBinsHI": 500, "GridDimensions": 1000, "LengthUnits": 6.6 * cm_per_kpc, \
              "TimeUnits": s_per_myr, "StopTime": 15.0, 
              "StartRadius": 0.001, "MaxHIIFraction": 0.9999, "dtDataDump": 1.0, "DataDumpName": 'dd', \
              "Isothermal": 0, "MultiSpecies": 0, "SecondaryIonization": 1, "CosmologicalExpansion": 0, \
              "DensityProfile": 0, "InitialDensity": 2e-4, "TemperatureProfile": 0, "InitialTemperature": 8e3, \
              "IonizationProfile": 1, "InitialHIIFraction": 1.2e-3, "SourceType": 1, "SourceLifetime": 500.0, \
              "SourceTemperature": 1e5, "DiscreteSpectrumMethod": 1, "DiscreteSpectrumSED": [16.74, 24.65, 34.49, 52.06], \
              "DiscreteSpectrumRelLum": [0.277, 0.335, 0.2, 0.188], "SpectrumMinEnergy": 0.1, "SpectrumMaxEnergy": 100,
              "Clump": 1, "ClumpPosition": 0.76, "ClumpOverdensity": 200, "ClumpRadius": 0.8 / 6.6, "ClumpTemperature": 40.,
              "SpectrumPhotonLuminosity": 3e51
             }               
             
            
        
    return pf    