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
           'Cosmology', 'Solver', 'Control']
for grp in pgroups:
    defaults.append('%sParameters()' % grp)
    
def SetAllDefaults():
    pf = {'ProblemType': 1}
    
    for pset in defaults:
        eval('pf.update(%s)' % pset)
        
    return pf
    
def SpectrumParameters(): 
    pf = \
        {
        "spectrum_multifreq": 0,
        "spectrum_multigrp": 0,
        "FrequencyAveragedCrossSections": 0,
        "FrequencyGroups": 1,
        "bands": [13.6, 24.6, 54.4],
        "spectrum_type": 1,
        "spectrum_fraction": 1,   
        "spectrum_alpha": 1.5,  
        "spectrum_Emin": 13.6,  
        "spectrum_Emax": 1e2,  
        "spectrum_EminNorm": 0.01,  
        "spectrum_EmaxNorm": 5e2,  
        "spectrum_qdot": 5e48,  
        "spectrum_N": 0,
        "spectrum_fcol": 1.7,
        "spectrum_file": 'None',
        "ForceIntegralTabulation": 0,
          
        "spectrum_E": [13.6],  
        "spectrum_LE": [1.],  
        
        "IntegralTable": 'None',  
        "RegenerateTable": 0,  
        "ColumnDensityBinsHI": 200,  
        "ColumnDensityBinsHeI": 100,  
        "ColumnDensityBinsHeII": 100,  
        "IonizedFractionBins": 20,
        "AgeBins": 20,
        }
        
    return pf       

def SourceParameters():
    pf = \
        {
        "source_N": 1,
        "source_files": 'None',
        "source_type": 1,  
        "source_temperature": 1e5,  
        "source_mass": 1e3,  
        "source_fduty": 1,
        "source_tbirth": 0,
        "source_lifetime": 1e10,  
        "source_eta": 0.1,
        "source_isco": 0,  
        "source_rmax": 1e3,
        "source_evolving": 0,
        "source_cX": 1.0,
        }
        
    return pf
    
def PhysicsParameters():
    pf = \
        {
        "PhotonConserving": 1,  
        #"LymanAlphaContinuum": 0,
        #"LymanAlphaInjection": 0,
        "Isothermal": 0,  
        #"MultiSpecies": 0,  
        #"CollisionalIonization": 1,  
        #"CollisionalExcitation": 1,  
        "SecondaryIonization": 0,  
        "SecondaryLymanAlpha": 0,
        #"ComptonHeating": 0,  
        #"FreeFreeEmission": 0,  
        "CosmologicalExpansion": 0,  
        #"InfiniteSpeedOfLight": 1,  
        #"PlaneParallelField": 0,  
        "ClumpingFactor": 1,
        "RecombinationMethod": 'B', 
        "SecondaryElectronDataFile": 'None',        
        }
        
    return pf    
    
def CosmologyParameters():
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
    
def ControlParameters():
    pf = \
        {
        "ParallelizationMethod": 1,  
        "Debug": 1,  
        "CurrentTime": 0.0,  
        "CurrentRedshift": 'None',
        "StopTime": 50.0,  
        "dtDataDump": 5.0,
        "dzDataDump": 0.0,  
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
        "OpticallyThinColumn": [15, 16, 16],
        "ExitAfterIntegralTabulation": 0,
        }
        
    return pf
    
def GridParameters():
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
    
def SolverParameters():    
    pf = \
        {
        "UseScipy": 1,
        "InterpolationMethod": 0,   
        "MaximumGlobalTimestep": 500,  
        "MinimumSpeciesFraction": 1e-8,  
        }
        
    return pf
    
