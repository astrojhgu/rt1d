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
        "IntegrationMethod": 0, \
        "InterpolationMethod": 0, \
        
        # ODE Solver
        "ODEAdaptiveStep": 1, \
        "ODEAdaptiveFreq": 1, \
        "ODEMinStep": 0.01, \
        "ODEMaxStep": 0.1, \
        "ODErtol": 1e-6, \
        "ODEatol": 1e-8, \
        "ODEmaxiter": 100, \
        "RootFinder": 0, \
          
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
        "GridDimensions": 1000, \
          
        # Units
        "LengthUnits": cm_per_kpc, \
        "TimeUnits": s_per_myr, \
          
        # Control parameters
        "CurrentTime": 0.0, \
        "StopTime": 50.0, \
        "GlobalTimestep": 0.05, \
        "AdaptiveTimestep": 0, \
        "TimestepSafetyFactor": 0.5, \
        "StartRadius": 0.001, \
        "dtDataDump": 1.0, \
        "DataDumpName": 'dd', \
        "SavePrefix": 'rt', \
        
        # Physics
        "SolveTemperatureEvolution": 1, \
        "MultiSpecies": 0, \
        "CollisionalIonization": 1, \
        "CollisionalExcitation": 1, \
        # Replace SecondaryelectronMethod with SecondaryIonization (can be greater than 1)
        "SecondaryIonization": 1, \
        "SecondaryElectronMethod": 1, \
        "ComptonCooling": 0, \
        "HubbleCooling": 0, \
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
