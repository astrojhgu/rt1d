"""
SetDefaultParameterValues.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-10-19.

Description: Defaults for all different kinds of parameters, now sorted
into groups.
     
"""

from .Constants import *

defaults = []
pgroups = ['Grid', 'Source', 'Spectrum', 'Physics', 
           'Cosmology',  'Solver', 'Control']
for grp in pgroups:
    defaults.append('SetDefault%sParameters()' % grp)
    
def SetAllDefaults():
    pf = {'ProblemType': 1}
    
    for pset in defaults:
        eval('pf.update(%s)' % pset)
        
    return pf
        
def SetDefaultSourceParameters():
    pf = \
        {
        "NumberOfSources": 1,
        "SourceParameters": 'None',
        "SourceType": 1,  
        "SourceTemperature": 1e5,  
        "SourceMass": 1e3,  
        "SourceLifetime": 1e10,  
        "SourceRadiativeEfficiency": 0.1,
        "SourceISCO": 0,  
        "SourceDiskMaxRadius": 1e3,
        "SourceTimeEvolution": 0,
        }
        
    return pf

def SetDefaultSpectrumParameters():
    pf = \
        {
        "DiscreteSpectrum": 0,
        "FrequencyAveragedCrossSections": 0,
        "FrequencyGroups": 1,
        "FrequencyBands": [13.6, 24.6, 54.4],
        "SpectrumType": 1, 
        "SpectrumFraction": 1,   
        "SpectrumPowerLawIndex": 1.5,  
        "SpectrumMinEnergy": 13.6,  
        "SpectrumMaxEnergy": 1e2,  
        "SpectrumMinNormEnergy": 0.01,  
        "SpectrumMaxNormEnergy": 5e2,  
        "SpectrumPhotonLuminosity": 5e48,  
        "SpectrumAbsorbingColumn": 0,  
        "SpectrumColorCorrectionFactor": 1.7,
        "SpectrumFile": 'None',
        "ForceIntegralTabulation": 0,
          
        "DiscreteSpectrumSED": [13.6],  
        "DiscreteSpectrumRelLum": [1.],  
        
        "IntegralTable": 'None',  
        "RegenerateTable": 0,  
        "ColumnDensityBinsHI": 200,  
        "ColumnDensityBinsHeI": 100,  
        "ColumnDensityBinsHeII": 100,  
        "IonizedFractionBins": 20,
        "AgeBins": 20,
        "ExitAfterIntegralTabulation": 0,  
        
        "SecondaryElectronDataFile": 'None',
        }
        
    return pf    

def SetDefaultPhysicsParameters():
    pf = \
        {
        "PhotonConserving": 1,  
        "Isothermal": 1,  
        "MultiSpecies": 0,  
        "CollisionalIonization": 1,  
        "CollisionalExcitation": 1,  
        "SecondaryIonization": 0,  
        "ComptonHeating": 0,  
        "FreeFreeEmission": 0,  
        "CosmologicalExpansion": 0,  
        "InfiniteSpeedOfLight": 1,  
        "PlaneParallelField": 0,  
        "ClumpingFactor": 1,
        "RecombinationMethod": 'B', 
        "SecondaryElectronDataFile": 'None',        
        }
        
    return pf
    
def SetDefaultCosmologyParameters():
    pf = \
        {
        "OmegaMatterNow": 0.272,  
        "OmegaBaryonNow": 0.044,  
        "OmegaLambdaNow": 0.728,  
        "HubbleParameterNow": 0.702,
        "PrimordialHeliumByMass": 0.2477,
        "CMBTemperatureNow": 2.725,
        }
        
    return pf
    
def SetDefaultControlParameters():
    pf = \
        {
        "ParallelizationMethod": 1,  
        "Debug": 1,  
        "CurrentTime": 0.0,  
        "CurrentRedshift": 'None',
        "StopTime": 50.0,  
        "dtDataDump": 5.0,  
        "LogarithmicDataDump": 0,
        "InitialLogDataDump": 1e-6,
        "NlogDataDumps": 100, 
        "DataDumpName": 'dd',  
        "OutputDirectory": '.',  
        "OutputFormat": 1,  
        "OutputTimestep": 1,  
        "OutputRates": 1,  
        "ProgressBar": 0,  
        "CurrentTimestep": 1e-8,  
        "InitialTimestep": 0,  
        "HIRestrictedTimestep": 1,  
        "HeIRestrictedTimestep": 0,  
        "HeIIRestrictedTimestep": 1,  
        "HeIIIRestrictedTimestep": 1,  
        "OpticalDepthDefiningIFront": [0.5, 0.5, 0.5],  
        "ElectronRestrictedTimestep": 0,
        "TemperatureRestrictedTimestep": 0,
        "LightCrossingTimeRestrictedTimestep": 0,
        "RedshiftRestrictedTimestep": 0,  
        "OnePhotonPackagePerCell": 0,  
        "MaxHIIChange": 0.05,  
        "MaxHeIChange": 0.05,  
        "MaxHeIIChange": 0.05,  
        "MaxHeIIIChange": 0.05,  
        "MaxElectronChange": 0.05,  
        "MaxTemperatureChange": 0.05,
        "MaxRedshiftStep": 1, 
        "AllowSmallTauApprox": 0,
        "OpticallyThinColumn": [15, 16, 0],
        }
        
    return pf
    
def SetDefaultGridParameters():
    pf = \
        {
        "GridDimensions": 100,  
        "LogarithmicGrid": 0,  
        "StartRadius": 0.01,  
        
        "InitialRedshift": 20.0, 
        "FinalRedshift": 6.,
        "DensityProfile": 0,  
        "TemperatureProfile": 0,  
        "InitialTemperature": 1e4,  
        "IonizationProfile": 0,  
        "InitialHIIFraction": 1e-4,  
        "Clump": 0,  
        "ClumpPosition": 0.1,  
        "ClumpRadius": 0.05,  
        "ClumpDensityProfile": 0,  
        "ClumpOverdensity": 100,  
        "ClumpTemperature": 100,  
          
        "DensityUnits": 1e-3 * m_H,
        "LengthUnits": cm_per_kpc,  
        "TimeUnits": s_per_myr,  
        }

    return pf
    
def SetDefaultSolverParameters():    
    pf = \
        {
        "InterpolationMethod": 0,  
        "ODEMinStep": 1e-8,  
        "ODEMaxStep": 1e-1,  
        "ODEMaxIter": 1e3,  
        "ODEAdaptiveStep": 1,  
        "ODEConservativeAdaptiveStep": 1,
        "ODEConservativeTolerance": 1e-2,
        "CheckForGoofiness": 1,  
        "MaximumGlobalTimestep": 500,  
        "MinimumSpeciesFraction": 1e-6,  
        }
        
    return pf
    
