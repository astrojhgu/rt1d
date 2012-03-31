"""

ProblemTypes.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Mar  7 15:53:15 2012

Description: Non-default parameter sets for certain test problems.

Note: Integer problem types imply use of a continuous SED, while their
non-integer counterparts imply a discrete SED, except for ProblemType = 1 = 1.1.
I generally (for clarity) order parameters from top to bottom in the following
way:
    Units, time/data dump interval
    Integral tabulation
    Initial conditions
    Source parameters
    Physics parameters

"""

import numpy as np

cm_per_kpc = 3.08568 * 10**21
s_per_myr = 365.25 * 24 * 3600 * 10**6

def ProblemType(ptype):
    """
    Storage bin for predefined problem types, 'pt's, like those used in the 
    radiative transfer comparison project ('RT06').
    """
    
    # RT06-1, RT1: Pure hydrogen, isothermal HII region expansion, monochromatic spectrum at 13.6 eV
    if ptype in [1, 1.1]:
        pf = {"ProblemType": 1, 
              "TabulateIntegrals": 0, 
              "LengthUnits": 6.6 * cm_per_kpc, 
              "StopTime": 500.0, 
              "dtDataDump": 10.0, 
              "MinimumSpeciesFraction": 1e-6, 
              "DensityProfile": 0,
              "InitialDensity": 1e-3, 
              "TemperatureProfile": 0, 
              "InitialTemperature": 1e4,
              "IonizationProfile": 0, 
              "InitialHIIFraction": 1.2e-3, 
              "SourceType": 0, 
              "SpectrumPhotonLuminosity": 5e48, 
              "DiscreteSpectrum": 1, 
              "DiscreteSpectrumSED": [13.6],
              "DiscreteSpectrumRelLum": [1.0],
              "CollisionalIonization": 0               
             }        
            
    # RT06-2: Pure hydrogen, HII region expansion, temperature evolution allowed, *continuous spectrum*
    if ptype == 2:
       pf = {"ProblemType": 2, 
             "LengthUnits": 6.6 * cm_per_kpc,
             "StopTime": 500.0,
             "dtDataDump": 10.0, 
             "ColumnDensityBinsHI": 500, 
             "MinimumSpeciesFraction": 1e-6, 
             "DensityProfile": 0, 
             "InitialDensity": 1e-3, 
             "TemperatureProfile": 0, 
             "InitialTemperature": 1e2,
             "IonizationProfile": 0, 
             "InitialHIIFraction": 1.2e-3, 
             "SourceType": 1, 
             "SpectrumPhotonLuminosity": 5e48, 
             "SpectrumMinEnergy": 13.6, 
             "SpectrumMaxEnergy": 100., \
             "SpectrumMinNormEnergy": 0.1, 
             "SpectrumMaxNormEnergy": 100., 
             "DiscreteSpectrum": 0,
             "Isothermal": 0 
            }
            
    # RT06-2: Pure hydrogen, HII region expansion, temperature evolution allowed, 4-bin spectrum of Wise & Abel (2011)
    if ptype == 2.1:
        pf = {"ProblemType": 2, 
             "LengthUnits": 6.6 * cm_per_kpc,
             "StopTime": 500.0,
             "dtDataDump": 10.0, 
             "TabulateIntegrals": 0,
             "MinimumSpeciesFraction": 1e-6, 
             "DensityProfile": 0, 
             "InitialDensity": 1e-3, 
             "TemperatureProfile": 0, 
             "InitialTemperature": 1e2,
             "IonizationProfile": 0, 
             "InitialHIIFraction": 1.2e-3, 
             "SourceType": 1, 
             "SpectrumPhotonLuminosity": 5e48, 
             "DiscreteSpectrum": 1,
             "DiscreteSpectrumSED": [16.74, 24.65, 34.49, 52.06], 
             "DiscreteSpectrumRelLum": [0.277, 0.335, 0.2, 0.188],
             "Isothermal": 0
            }
    
    # RT06-2: Pure hydrogen, HII region expansion, temperature evolution allowed, 4-bin spectrum of Mirocha et al. (2012)        
    if ptype == 2.2:
        pf = {"ProblemType": 2, 
             "LengthUnits": 6.6 * cm_per_kpc,
             "StopTime": 500.0,
             "dtDataDump": 10.0, 
             "TabulateIntegrals": 0,
             "MinimumSpeciesFraction": 1e-6, 
             "DensityProfile": 0, 
             "InitialDensity": 1e-3, 
             "TemperatureProfile": 0, 
             "InitialTemperature": 1e2,
             "IonizationProfile": 0, 
             "InitialHIIFraction": 1.2e-3, 
             "SourceType": 1, 
             "SpectrumPhotonLuminosity": 5e48, 
             "DiscreteSpectrum": 1,
             "DiscreteSpectrumSED": [18.29, 31.46, 49.13, 77.23], 
             "DiscreteSpectrumRelLum": [0.24, 0.35, 0.23, 0.06],
             "Isothermal": 0
            }    
    
    
    
             
    # RT06-3: I-front trapping in a dense clump and the formation of a shadow - continuous BB spectrum
    if ptype == 3:
        pf = {"ProblemType": 3,  
              "LengthUnits": 6.6 * cm_per_kpc,
              "GridDimensions": 200,
              "StopTime": 15.0, 
              "dtDataDump": 1.0, 
              "Isothermal": 0, 
              "DensityProfile": 0, 
              "InitialDensity": 2e-4, 
              "TemperatureProfile": 0, 
              "InitialTemperature": 8e3, 
              "IonizationProfile": 0, 
              "InitialHIIFraction": 1e-5, 
              "SourceType": 1, 
              "SpectrumPhotonLuminosity": 1e6, 
              "SpectrumMinEnergy": 13.6, 
              "SpectrumMaxEnergy": 100., \
              "SpectrumMinNormEnergy": 0.1, 
              "SpectrumMaxNormEnergy": 100., 
              "DiscreteSpectrum": 0,
              "Clump": 1, 
              "ClumpPosition": 5.0 / 6.6, 
              "ClumpOverdensity": 200., 
              "ClumpRadius": 0.8 / 6.6,
              "ClumpTemperature": 40., 
              "PlaneParallelField": 1
             }          
    
    # RT06-3: I-front trapping in a dense clump and the formation of a shadow
    if ptype == 3.1:
        pf = {"ProblemType": 3.1,  
              "LengthUnits": 6.6 * cm_per_kpc,
              "GridDimensions": 200,
              "StopTime": 15.0, 
              "dtDataDump": 1.0, 
              "Isothermal": 0, 
              "DensityProfile": 0, 
              "InitialDensity": 2e-4, 
              "TemperatureProfile": 0, 
              "InitialTemperature": 8e3, 
              "IonizationProfile": 0, 
              "InitialHIIFraction": 1e-5, 
              "SourceType": 0, 
              "SpectrumPhotonLuminosity": 1e6, 
              "DiscreteSpectrum": 1, 
              "DiscreteSpectrumSED": [16.74, 24.65, 34.49, 52.06], 
              "DiscreteSpectrumRelLum": [0.277, 0.335, 0.2, 0.188], 
              "Clump": 1, 
              "ClumpPosition": 5.0 / 6.6, 
              "ClumpOverdensity": 200., 
              "ClumpRadius": 0.8 / 6.6,
              "ClumpTemperature": 40.,
              "PlaneParallelField": 1
             }   
             
    # X-ray source, helium included, continuous spectrum       
    if ptype == 5:
        pf = {"ProblemType": 5, "InterpolationMethod": 0, \
              "ColumnDensityBinsHI": 100, "ColumnDensityBinsHeI": 50, "ColumnDensityBinsHeII": 50, 
              "GridDimensions": 100, "LengthUnits": 1000 * cm_per_kpc, \
              "TimeUnits": s_per_myr, "CurrentTime": 0.0, "StopTime": 45.0, 
              "StartRadius": 0.01, "dtDataDump": 1, "DataDumpName": 'dd', \
              "Isothermal": 0, "MultiSpecies": 1, "SecondaryIonization": 1, "CosmologicalExpansion": 0, \
              "DensityProfile": 1, "InitialRedshift": 10, "TemperatureProfile": 0, "InitialTemperature": 1e2, \
              "IonizationProfile": 0, "InitialHIIFraction": 1e-4, "SourceType": 3, "SourceLifetime": 1e10, \
              "SourceMass": 1e6, "SourceRadiativeEfficiency": 0.1, "SpectrumPowerLawIndex": 1.5, \
              "SpectrumMinEnergy": 1e2, "SpectrumMaxEnergy": 1e4, "SpectrumMinNormEnergy": 1e2, "SpectrumMaxNormEnergy": 1e4 
             }      
             
    # X-ray source, helium included, discrete spectrum         
    if ptype == 5.1:
        pf = {"ProblemType": 5.1, "InterpolationMethod": 0, "TabulateIntegrals": 0, \
              "ColumnDensityBinsHI": 100, "ColumnDensityBinsHeI": 50, "ColumnDensityBinsHeII": 50, 
              "GridDimensions": 100, "LengthUnits": 1000 * cm_per_kpc, \
              "TimeUnits": s_per_myr, "CurrentTime": 0.0, "StopTime": 45.0, 
              "StartRadius": 0.01, "dtDataDump": 1, "DataDumpName": 'dd', \
              "Isothermal": 0, "MultiSpecies": 1, "SecondaryIonization": 1, "CosmologicalExpansion": 0, \
              "DensityProfile": 1, "InitialRedshift": 10, "TemperatureProfile": 0, "InitialTemperature": 1e2, \
              "IonizationProfile": 0, "InitialHIIFraction": 1e-4, "SourceType": 3, "SourceLifetime": 1e10, \
              "DiscreteSpectrumSED": [500.], "DiscreteSpectrumRelLum": [1.], "DiscreteSpectrum": 1,\
              "SourceMass": 1e6, "SourceRadiativeEfficiency": 0.1
             }                                 
             
        
    return pf    