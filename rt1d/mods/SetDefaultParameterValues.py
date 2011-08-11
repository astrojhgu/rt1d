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

from rt1d.mods.Constants import *

def SetDefaultParameterValues():
    pf = \
       {
    
        # Override
        "ProblemType": 0, \
          
        # Integration
        "InterpolationMethod": 0, \
        
        # ODE Solver
        "ODEAdaptiveStep": 1, \
        "ODEMinStep": 1e-8, \
        "ODEMaxStep": 1e-1, \
        "ODErtol": 1e-3, \
        "ODEatol": 1e-6, \
        "ODEmaxiter": 10000, \
        "RootFinder": 0, \
        
        "ParallelizationMethod": 0, \
          
        # Integral tabulation
        "HIColumnMin": 1e15, \
        "HIColumnMax": 1e22, \
        "HeIColumnMin": 1e17, \
        "HeIColumnMax": 1e22, \
        "HeIIColumnMin": 1e18, \
        "HeIIColumnMax": 1e23, \
        "ColumnDensityBinsHI": 50, \
        "ColumnDensityBinsHeI": 10, \
        "ColumnDensityBinsHeII": 10, \
        "ExitAfterIntegralTabulation": 0, \
          
        # Grid parameters
        "GridDimensions": 100, \
          
        # Units
        "LengthUnits": cm_per_kpc, \
        "TimeUnits": s_per_myr, \
          
        # Control parameters
        "CurrentTime": 0.0, \
        "StopTime": 50.0, \
        "StartRadius": 0.01, \
        "dtDataDump": 1.0, \
        "DataDumpName": 'dd', \
        "OutputDirectory": './', \
        "OutputTimestep": 1, \
        "ProgressBar": 1, \
        "InitialTimestep": 1e-8, \
        "CurrentTimestep": 1e-8, \
        "HIIRestrictedTimestep": 1, \
        "HeIIRestrictedTimestep": 0, \
        "MaxHIIChange": 0.05, \
        "MaxHeIIChange": 0.05, \
        
        # Physics
        "Isothermal": 0, \
        "MultiSpecies": 0, \
        "CollisionalIonization": 1, \
        "CollisionalExcitation": 1, \
        "SecondaryIonization": 1, \
        "ComptonCooling": 1, \
        "CosmologicalExpansion": 0, \
        
        # Initial conditions
        "DensityProfile": 0, \
        "InitialDensity": 0, \
        "TemperatureProfile": 0, \
        "InitialTemperature": 100, \
        "IonizationProfile": 0, \
        "InitialHIIFraction": 1e-4, \
        "Clump": 0, \
        "ClumpPosition": 0.1, \
        "ClumpRadius": 0.05, \
        "ClumpDensityProfile": 0, \
        "ClumpOverdensity": 100, \
        "ClumpTemperature": 100, \
          
        # Source parameters
        "SourceType": 0, \
        "SourceTemperature": 1e5, \
        "SourceRadius": 1.0, \
        "SourceMass": 1e3, \
        "SourceLifetime": 1e10, \
        "SourceRadiativeEfficiency": 0.1, \
          
        # Spectral parameters
        "DiscreteSpectrum": 0, \
        "SpectrumPowerLawIndex": 1.5, \
        "SpectrumMinEnergy": 100, \
        "SpectrumMaxEnergy": 1e4, \
        "SpectrumMinNormEnergy": 100, \
        "SpectrumMaxNormEnergy": 1e4, \
        "SpectrumPhotonLuminosity": 5e48, \
          
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
