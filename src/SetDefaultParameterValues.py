"""
SetDefaultParameterValues.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-10-19.

Description: Complete parameter list with default values.  Stored as a python dictionary, read in when we initialize
the parameter space.
     
To do: New naming convention:
     -Source properties prefaced with "Source": SourceType, SourceMass, SourceTemperature, etc.
     -Spectral properties labeled with "Spectrum", SpectrumMinEnergy, SpectrumPowerLawIndex, etc.
     -Properties only applicabale for discretized spectra labeled with "DiscreteSpectrum", DiscreteSpectrumMinEnergy, DiscreteSpectrumBins, etc.
     
"""

from Constants import *

def SetDefaultParameterValues():
    pf = \
       {
    
        # Override
        "ProblemType": 0, \
          
        # Integration
        "IntegrationMethod": 0, \
        "InterpolationMethod": 1, \
        
        # ODE Solver
        "ODEAdaptiveStep": 1, \
        "ODEMinStep": 0.01, \
        "ODEMaxStep": 0.1, \
        "ODErtol": 1e-6, \
        "ODEatol": 1e-8, \
        "ODEmaxiter": 100, \
        "RootFinder": 0, \
          
        # Integral tabulation
        "ColumnDensityBinsHI": 100, \
        "ColumnDensityBinsHeI": 50, \
        "ColumnDensityBinsHeII": 50, \
        "ExitAfterIntegralTabulation": 0, \
          
        # Grid parameters
        "GridDimensions": 1000, \
          
        # Units
        "LengthUnits": cm_per_kpc, \
        "TimeUnits": s_per_myr, \
          
        # Control parameters
        "CurrentTime": 0.0, \
        "StopTime": 50.0, \
        "InitialTimestep": 0.1, \
        "AdaptiveTimestep": 0, \
        "TimestepSafetyFactor": 0.5, \
        "StartRadius": 0.001, \
        "MaxHIIFraction": 0.9999, \
        "dtDataDump": 1.0, \
        "DataDumpName": 'dd', \
        "SavePrefix": 'rt', \
        
        # Physics
        "SolveTemperatureEvolution": 1, \
        "MultiSpecies": 0, \
        "SecondaryElectronMethod": 1, \
        "ComptonCooling": 1, \
        "CosmologicalExpansion": 0, \
        
        # Initial conditions
        "DensityProfile": 0, \
        "InitialDensity": 0, \
        "TemperatureProfile": 0, \
        "InitialTemperature": 100, \
        "IonizationProfile": 0, \
        "InitialHIIFraction": 1e-4, \
          
        # Source parameters
        "SourceType": 0, \
        "SourceTemperature": 1e4, \
        "SourceRadius": 1.0, \
        "SourceMass": 1e3, \
        "SourceLifetime": 50.0, \
        "SourceRadiativeEfficiency": 0.1, \
          
        # Spectral parameters
        "SpectrumPowerLawIndex": 1.0, \
        "SpectrumMinEnergy": 100, \
        "SpectrumMaxEnergy": 1e4, \
        "SpectrumMinNormEnergy": 100, \
        "SpectrumMaxNormEnergy": 1e4, \
        "SpectrumPhotonLuminosity": 1e54, \
          
        # Discretization parameters
        "DiscreteSpectrumMethod": 0, \
        "DiscreteSpectrumMinEnergy": 100., \
        "DiscreteSpectrumMaxEnergy": 1e4, \
        "DiscreteSpectrumNumberOfBins": 20, \
        "DiscreteSpectrumBinEdges": [13.6, 20, 30, 40], \
        "DiscreteSpectrumSED": [16.74, 24.65, 34.49, 52.06], \
        "DiscreteSpectrumRelLum": [0.277, 0.335, 0.2, 0.188], \
          
        # Cosmological Parameters (WMAP 7)
        "InitialRedshift": 20.0, \
        "OmegaMatterNow": 0.272, \
        "OmegaBaryonNow": 0.044, \
        "OmegaLambdaNow": 0.728, \
        "HubbleParameterNow": 0.702
          
        }

    return pf
