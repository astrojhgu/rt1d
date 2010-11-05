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
          "GridDimensions": 1000, \
          
          # Packages
          "UseScipy": 1, \
          "IntegrationMethod": 0, \
          "InterpolationMethod": 1, \
          
          # Integral tabulation
          "ColumnDensityBins": 50, \
          
          # Units
          "LengthUnits": cm_per_kpc, \
          "TimeUnits": s_per_myr, \
          
          # Control parameters
          "InitialTimestep": 0.01, \
          "StopTime": 50.0, \
          "dtDataDump": 5.0, \
          "DataDumpName": 'dd', \
          "SavePrefix": 'rt', \
          "StartRadius": 0.01, \
          
          # General parameters
          "MultiSpecies": 0, \
          
          # Initial conditions
          "DensityProfile": 0, \
          "InitialDensity": 0, \
          "TemperatureProfile": 0, \
          "InitialTemperature": 100, \
          "IonizationProfile": 0, \
          "InitialHIIFraction": 1e-4, \
          
          # Source parameters
          "SourceSpectrum": 0, \
          "SourceTemperature": 1e4, \
          "SourceRadius": 1.0, \
          "SourceMass": 1e3, \
          "SourcePowerLawIndex": 1.0, \
          "SourceLifetime": 50.0, \
          "SourceMaxEnergy": 1e2, \
          "SourceMinEnergy": 1, \
          
          # Cosmological Parameters (WMAP 7)
          "InitialRedshift": 20.0, \
          "CosmologicalExpansion": 0, \
          "OmegaMatterNow": 0.272, \
          "OmegaBaryonNow": 0.044, \
          "OmegaLambdaNow": 0.728, \
          "HubbleParameterNow": 0.702
          
         }

    return pf