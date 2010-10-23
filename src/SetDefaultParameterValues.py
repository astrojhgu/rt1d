"""
SetDefaultParameterValues.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-08-19.

Description: Complete parameter list with default values.  Stored as a python dictionary, read in when we initialize
the parameter space.
     
"""

from Constants import *

def SetDefaultParameterValues():
    pf = {
          # Grid parameters
          "GridDimensions": 1024, \
          
          # Units
          "LengthUnits": cm_per_kpc, \
          "TimeUnits": s_per_myr, \
          "DensityUnits": 1e-30, \
          
          # Control parameters
          "InitialTimestep": 0.01, \
          "StopTime": 50.0, \
          "dtDataDump": 1.0, \
          "DataDumpName": 'dd', \
          "SavePrefix": 'rt', \
          
          # Initial conditions
          "DensityProfile": 1, \
          "InitialDensity": 0, \
          "TemperatureProfile": 0, \
          "InitialTemperature": 100, \
          "IonizationProfile": 0, \
          "InitialHIIFraction": 1e-4, \
          
          # Source parameters
          "SourceSpectrum": 0, \
          "SourceLuminosity": 1e40, \
          "SourceLifetime": 50.0, \
          
          # Cosmological Parameters (WMAP 7)
          "InitialRedshift": 20.0, \
          "CosmologicalExpansion": 0, \
          "OmegaMatterNow": 0.272, \
          "OmegaBaryonNow": 0.044, \
          "OmegaLambdaNow": 0.728, \
          "HubbleParameterNow": 0.702
          
         }

    return pf