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
    
More notes:
10 + ProblemType = same as ProblemType but MultiSpecies = 1    

"""

import numpy as np
from ..physics.Constants import m_H, cm_per_kpc, cm_per_mpc, s_per_myr

def ProblemType(ptype):
    """
    Storage bin for predefined problem types, like those used in the 
    radiative transfer comparison project ('RT06').
    """
    
    # RT06-0.3, Single zone ionization/heating, then source switches off.
    if ptype == 0:
        pf = {"problem_type": 0, 
              "TabulateIntegrals": 1, 
              "LengthUnits": 1e-4 * cm_per_kpc, # 100 milliparsecs 
              "GridDimensions": 1, 
              "StartRadius": 0.99, # cell = 1 milliparsec across
              "StopTime": 10, 
              "LogarithmicDataDump": 1,
              "InitialLogDataDump": 1e-12,
              "ElectronRestrictedTimestep": 1,
              "NlogDataDumps": 100,
              "dtDataDump": 0.1, 
              "ODEMinStep": 1e-15,
              "MinimumSpeciesFraction": 1e-9, 
              "ColumnDensityBinsHI": 100,
              "DensityUnits": m_H,
              "DensityProfile": 0,
              "InitialDensity": 1., 
              "TemperatureProfile": 0, 
              "InitialTemperature": 1e2,
              "IonizationProfile": 0, 
              "InitialHIIFraction": 1e-6, 
              "SourceType": 1, 
              "SpectrumType": 1,
              "SourceLifetime": 0.5,
              "OpticalDepthDefiningIFront": [0, 0, 0],
              "SpectrumPhotonLuminosity": 1e12, 
              "SpectrumMinEnergy": 13.6, 
              "SpectrumMaxEnergy": 100.,
              "SpectrumMinNormEnergy": 0.1, 
              "SpectrumMaxNormEnergy": 100., 
              "DiscreteSpectrum": 0,
              "Isothermal": 0,
              "PlaneParallelField": 1              
             }  
    
    # Single-zone, cosmological expansion test         
    if ptype == 0.1:
        pf = {"ProblemType": 0.1, 
              "CosmologicalExpansion": 1,
              "TabulateIntegrals": 0, 
              "LengthUnits": 1e-4 * cm_per_kpc, # 100 milliparsecs 
              "GridDimensions": 1, 
              "StartRadius": 0.99, # cell = 1 milliparsec across
              "StopTime": 800, 
              "dtDataDump": 20, 
              "ODEMinStep": 1e-15,
              "InitialRedshift": 400., 
              "FinalRedshift": 6,
              "IonizationProfile": 0, 
              "InitialHIIFraction": 1e-4, 
              "SourceType": 0, 
              "SourceLifetime": 0,
              "DiscreteSpectrum": 1,
              "Isothermal": 0,
              "OpticalDepthDefiningIFront": [0, 0, 0]
             }  
                       
    
    # RT06-1, RT1: Pure hydrogen, isothermal HII region expansion, monochromatic spectrum at 13.6 eV
    if ptype in [1, 1.1]:
        pf = {"problem_type": 1, 
              "density_units": 1e-3 * m_H,
              "length_units": 6.6 * cm_per_kpc, 
              "stop_time": 500.0, 
              "dtDataDump": 10.0, 
              "isothermal": 1,
              "species": [1],
              "initial_temperature": 1e4,
              "initial_ionization": [1.2e-3], 
              "source_type": 0, 
              "spectrum_qdot": 5e48, 
              "spectrum_E": [13.6],
              "spectrum_LE": [1.0],
             }
            
    # RT06-2: Pure hydrogen, HII region expansion, temperature evolution allowed, *continuous spectrum*
    if ptype in [2, 2.1]:
       pf = {"problem_type": 2, 
             "density_units": 1e-3 * m_H,
             "length_units": 6.6 * cm_per_kpc, 
             "stop_time": 500.0, 
             "dtDataDump": 10.0, 
             "isothermal": 0,
             "restricted_timestep": ['ions', 'energy'],
             "species": [1],
             "secondary_ionization": 0,
             "initial_temperature": 1e2,
             "initial_ionization": [1.2e-3], 
             "source_type": 0, 
             "spectrum_qdot": 5e48, 
             "spectrum_E": [17.98, 31.15, 49.09, 76.98],
             "spectrum_LE": [0.23, 0.36, 0.24, 0.06],
            }

            
    # RT06-2: Hydrogen + helium, I-front expansion, temperature evolution allowed, *continuous spectrum*
    if ptype == 12:
       pf = {"ProblemType": 12, 
             "DensityUnits": 2.2245547230998267e-27,
             "LengthUnits": 6.6 * cm_per_kpc,
             "StopTime": 500.0,
             "dtDataDump": 10.0, 
             "OutputRates": 1,
             "MinimumSpeciesFraction": 1e-6, 
             "DensityProfile": 0, 
             "TemperatureProfile": 0, 
             "InitialTemperature": 1e2,
             "IonizationProfile": 0, 
             "InitialHIIFraction": 1.2e-3, 
             "MultiSpecies": 1,
             "SourceType": 1, 
             "SpectrumType": 1,
             "SpectrumPhotonLuminosity": 5e48, 
             "SpectrumMinEnergy": 13.6, 
             "SpectrumMaxEnergy": 100.,
             "SpectrumMinNormEnergy": 0.1, 
             "SpectrumMaxNormEnergy": 100., 
             "DiscreteSpectrum": 0,
             "Isothermal": 0 
            }        
            
    # RT06-2: Pure hydrogen, HII region expansion, temperature evolution allowed, 4-bin spectrum of Wise & Abel (2011)
    if ptype == 2.1:
        pf = {"ProblemType": 2, 
              "DensityUnits": 1e-3 * m_H,
              "LengthUnits": 6.6 * cm_per_kpc,
              "StopTime": 500.0,
              "dtDataDump": 10.0, 
              "TabulateIntegrals": 0,
              "MinimumSpeciesFraction": 1e-6, 
              "DensityProfile": 0, 
              "TemperatureProfile": 0, 
              "InitialTemperature": 1e2,
              "IonizationProfile": 0, 
              "InitialHIIFraction": 1.2e-3, 
              "SourceType": 1, 
              "SpectrumType": 1,
              "SpectrumPhotonLuminosity": 5e48, 
              "DiscreteSpectrum": 1,
              "DiscreteSpectrumSED": [16.74, 24.65, 34.49, 52.06], 
              "DiscreteSpectrumRelLum": [0.277, 0.335, 0.2, 0.188],
              "Isothermal": 0
             }
    
    # RT06-2: Pure hydrogen, HII region expansion, temperature evolution allowed, 4-bin spectrum of Mirocha et al. (2012)        
    if ptype == 2.2:
        pf = {"ProblemType": 2, 
              "DensityUnits": 1e-3 * m_H,
              "LengthUnits": 6.6 * cm_per_kpc,
              "StopTime": 500.0,
              "dtDataDump": 10.0, 
              "TabulateIntegrals": 0,
              "MinimumSpeciesFraction": 1e-6, 
              "DensityProfile": 0, 
              "TemperatureProfile": 0, 
              "InitialTemperature": 1e2,
              "IonizationProfile": 0, 
              "InitialHIIFraction": 1.2e-3, 
              "SourceType": 1, 
              "SpectrumType": 1,
              "SpectrumPhotonLuminosity": 5e48, 
              "DiscreteSpectrum": 1,
              "DiscreteSpectrumSED": [18.29, 31.46, 49.13, 77.23], 
              "DiscreteSpectrumRelLum": [0.24, 0.35, 0.23, 0.06],
              "Isothermal": 0
            }    
                 
    # RT06-3: I-front trapping in a dense clump and the formation of a shadow - continuous BB spectrum
    if ptype == 3:
        pf = {"ProblemType": 3,  
              "DensityUnits": 2e-4 * m_H,
              "LengthUnits": 6.6 * cm_per_kpc,
              "GridDimensions": 200,
              "StopTime": 15.0, 
              "dtDataDump": 1.0,
              "MaximumGlobalTimestep": 0.1,  
              "Isothermal": 0, 
              "DensityProfile": 0, 
              "TemperatureProfile": 0, 
              "InitialTemperature": 8e3, 
              "IonizationProfile": 0, 
              "InitialHIIFraction": 1e-5, 
              "SourceType": 1, 
              "SpectrumType": 1,
              "SpectrumPhotonLuminosity": 1e6, 
              "SpectrumMinEnergy": 13.6, 
              "SpectrumMaxEnergy": 100., 
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
              "DensityUnits": 2e-4 * m_H,
              "LengthUnits": 6.6 * cm_per_kpc,
              "GridDimensions": 200,
              "StopTime": 15.0, 
              "dtDataDump": 1.0, 
              "Isothermal": 0, 
              "DensityProfile": 0, 
              "TemperatureProfile": 0, 
              "InitialTemperature": 8e3, 
              "IonizationProfile": 0, 
              "InitialHIIFraction": 1e-5, 
              "SourceType": 0, 
              "SpectrumType": 1,
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
        pf = {"ProblemType": 5, 
              "LengthUnits": cm_per_mpc,
              "GridDimensions": 100, 
              "StopTime": 45.0, 
              "StartRadius": 0.01, 
              "dtDataDump": 1, 
              "Isothermal": 0, 
              "MultiSpecies": 1, 
              "SecondaryIonization": 1, 
              "DensityProfile": 1, 
              "InitialRedshift": 10, 
              "TemperatureProfile": 0, 
              "InitialTemperature": 1e2,
              "IonizationProfile": 0, 
              "InitialHIIFraction": 1e-4, 
              "SourceType": 3,
              "SourceMass": 1e6, 
              "SourceRadiativeEfficiency": 0.1, 
              "SpectrumType": 4,
              "SpectrumPowerLawIndex": 1.5,
              "SpectrumMinEnergy": 1e2, 
              "SpectrumMaxEnergy": 1e4, 
              "SpectrumMinNormEnergy": 1e2, 
              "SpectrumMaxNormEnergy": 1e4,
              "ColumnDensityBinsHI": 200, 
              "ColumnDensityBinsHeI": 100, 
              "ColumnDensityBinsHeII": 100
             }      
    # X-ray source, helium included, discrete spectrum       
    if ptype == 5.1:
        pf = {"ProblemType": 5.1, 
              "LengthUnits": cm_per_mpc,
              "GridDimensions": 100, 
              "StopTime": 45.0, 
              "StartRadius": 0.01, 
              "dtDataDump": 1, 
              "Isothermal": 0, 
              "MultiSpecies": 1, 
              "SecondaryIonization": 1, 
              "DensityProfile": 1, 
              "InitialRedshift": 10, 
              "TemperatureProfile": 0, 
              "InitialTemperature": 1e2,
              "IonizationProfile": 0, 
              "InitialHIIFraction": 1e-4, 
              "SourceType": 3,
              "SourceMass": 1e6, 
              "SourceRadiativeEfficiency": 0.1, 
              "SpectrumType": 4,
              "SpectrumPowerLawIndex": 1.5, 
              "SpectrumMinEnergy": 1e2, 
              "SpectrumMaxEnergy": 1e4, 
              "SpectrumMinNormEnergy": 1e2, 
              "SpectrumMaxNormEnergy": 1e4,
              "ColumnDensityBinsHI": 200, 
              "ColumnDensityBinsHeI": 100, 
              "ColumnDensityBinsHeII": 100,
              "DiscreteSpectrum": 1,
              "DiscreteSpectrumSED": [500.],
              "DiscreteSpectrumRelLum": [1.0]
             }      
             
        
    return pf    