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
        
        # This will prevent crashes if there is not a blank line at the end of the parameter file
        if line[-1] != '\n': line += '\n'
        
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
            if parval.replace('_', '').replace('.', '').strip().isalnum(): 
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
    Storage bin for predefined problem types, 'pt's, like those used in the radiative transfer comparison project ('RT06').
    """
    
    # RT06-1, RT1: Pure hydrogen, isothermal HII region expansion, monochromatic spectrum at 13.6 eV
    if pt == 1:
        pf = {"ProblemType": 1, "InterpolationMethod": 0, \
              "ColumnDensityBinsHI": 500, "GridDimensions": 100, "LengthUnits": 6.6 * cm_per_kpc, \
              "TimeUnits": s_per_myr, "CurrentTime": 0.0, "StopTime": 500.0, \
              "StartRadius": 0.01, "dtDataDump": 5.0, "DataDumpName": 'dd', \
              "Isothermal": 1, "MultiSpecies": 0, "SecondaryIonization": 0, "CosmologicalExpansion": 0, \
              "DensityProfile": 0, "InitialDensity": 1e-3, "TemperatureProfile": 0, "InitialTemperature": 1e4, \
              "IonizationProfile": 1, "InitialHIIFraction": 1.2e-3, "SourceType": 0, "SourceLifetime": 1e10, \
              "SpectrumPhotonLuminosity": 5e48, "DiscreteSpectrum": 1, "DiscreteSpectrumSED": [13.6], \
              "SpectrumMinEnergy": 0.1, "SpectrumMaxEnergy": 100, "CollisionalIonization": 0, "DiscreteSpectrumRelLum": [1.0]
             }        

    # RT06-2: Pure hydrogen, HII region expansion, temperature evolution allowed, 4-bin spectrum of Wise & Abel (2011)
    if pt == 2:
       pf = {"ProblemType": 2, "InterpolationMethod": 0, \
             "ColumnDensityBinsHI": 500, "GridDimensions": 100, "LengthUnits": 6.6 * cm_per_kpc, \
             "TimeUnits": s_per_myr, "CurrentTime": 0.0, "StopTime": 500.0, \
             "StartRadius": 0.01, "dtDataDump": 5.0, "DataDumpName": 'dd', \
             "Isothermal": 0, "MultiSpecies": 0, "SecondaryIonization": 1, "CosmologicalExpansion": 0, \
             "DensityProfile": 0, "InitialDensity": 1e-3, "TemperatureProfile": 0, "InitialTemperature": 1e2, \
             "IonizationProfile": 1, "InitialHIIFraction": 0, "SourceType": 1, "SourceLifetime": 1e10, \
             "SpectrumPhotonLuminosity": 5e48, "DiscreteSpectrum": 1, \
             "DiscreteSpectrumSED": [16.74, 24.65, 34.49, 52.06], "DiscreteSpectrumRelLum": [0.277, 0.335, 0.2, 0.188]
            } 
            
    # RT06-2: Pure hydrogen, HII region expansion, temperature evolution allowed, *continuous spectrum*
    if pt == 2.1:
       pf = {"ProblemType": 2.1, "InterpolationMethod": 0, \
             "ColumnDensityBinsHI": 500, "GridDimensions": 100, "LengthUnits": 6.6 * cm_per_kpc, \
             "TimeUnits": s_per_myr, "CurrentTime": 0.0, "StopTime": 500.0, \
             "StartRadius": 0.01, "dtDataDump": 5.0, "DataDumpName": 'dd', \
             "Isothermal": 0, "MultiSpecies": 0, "SecondaryIonization": 0, "CosmologicalExpansion": 0, \
             "DensityProfile": 0, "InitialDensity": 1e-3, "TemperatureProfile": 0, "InitialTemperature": 1e2, \
             "IonizationProfile": 1, "InitialHIIFraction": 0, "SourceType": 1, "SourceLifetime": 1e10, \
             "SpectrumPhotonLuminosity": 5e48, "SpectrumMinEnergy": 13.6, "SpectrumMaxEnergy": 100., \
             "SpectrumMinNormEnergy": 13.6, "SpectrumMaxNormEnergy": 100., "HIColumnMin": 1e16, \
             "HIColumnMax": 1e20, "DiscreteSpectrum": 0
            }
    
    # RT06-3: I-front trapping in a dense clump and the formation of a shadow
    if pt == 3:
        pf = {"ProblemType": 3, "InterpolationMethod": 0, \
              "ColumnDensityBinsHI": 500, "GridDimensions": 100, "LengthUnits": 6.6 * cm_per_kpc, \
              "TimeUnits": s_per_myr, "CurrentTime": 0.0, "StopTime": 15.0, 
              "StartRadius": 0.01, "dtDataDump": 1.0, "DataDumpName": 'dd', \
              "Isothermal": 0, "MultiSpecies": 0, "SecondaryIonization": 0, "CosmologicalExpansion": 0, \
              "DensityProfile": 0, "InitialDensity": 2e-4, "TemperatureProfile": 0, "InitialTemperature": 8e3, \
              "IonizationProfile": 1, "InitialHIIFraction": 0, "SourceType": 1, "SourceLifetime": 1e10, \
              "DiscreteSpectrumSED": [16.74, 24.65, 34.49, 52.06], \
              "DiscreteSpectrum": 1, "SpectrumPhotonLuminosity": 1e6, \
              "DiscreteSpectrumRelLum": [0.277, 0.335, 0.2, 0.188], \
              "Clump": 1, "ClumpPosition": 0.76, "ClumpOverdensity": 200, "ClumpRadius": 0.8 / 6.6, \
              "ClumpTemperature": 40, "PlaneParallelField": 1
             }   
    
    # X-ray source, hydrogen only       
    if pt == 4:
        pf = {"ProblemType": 4, "InterpolationMethod": 0, \
              "ColumnDensityBinsHI": 100, "ColumnDensityBinsHeI": 0, "ColumnDensityBinsHeII": 0, 
              "GridDimensions": 100, "LengthUnits": 100 * cm_per_kpc, \
              "TimeUnits": s_per_myr, "CurrentTime": 0.0, "StopTime": 10.0, 
              "StartRadius": 0.01, "dtDataDump": 1, "DataDumpName": 'dd', \
              "Isothermal": 0, "MultiSpecies": 0, "SecondaryIonization": 1, "CosmologicalExpansion": 0, \
              "DensityProfile": 1, "InitialRedshift": 20, "TemperatureProfile": 0, "InitialTemperature": 1e2, \
              "IonizationProfile": 1, "InitialHIIFraction": 1e-4, "SourceType": 3, "SourceLifetime": 1e10, \
              "SourceMass": 1e6, "SourceRadiativeEfficiency": 0.1, "SpectrumPowerLawIndex": 1.5
             }      
             
    # X-ray source, hydrogen only, discrete spectrum         
    if pt == 4.1:
        pf = {"ProblemType": 4.1, "InterpolationMethod": 0, \
              "ColumnDensityBinsHI": 50, "ColumnDensityBinsHeI": 0, "ColumnDensityBinsHeII": 0, 
              "GridDimensions": 100, "LengthUnits": 100 * cm_per_kpc, \
              "TimeUnits": s_per_myr, "CurrentTime": 0.0, "StopTime": 10.0, 
              "StartRadius": 0.01, "dtDataDump": 1, "DataDumpName": 'dd', \
              "Isothermal": 0, "MultiSpecies": 0, "SecondaryIonization": 1, "CosmologicalExpansion": 0, \
              "DensityProfile": 1, "InitialRedshift": 20, "TemperatureProfile": 0, "InitialTemperature": 1e2, \
              "IonizationProfile": 1, "InitialHIIFraction": 1e-4, "SourceType": 3, "SourceLifetime": 1e10, \
              "DiscreteSpectrumSED": [500.], "DiscreteSpectrumRelLum": [1.], "DiscreteSpectrum": 1,\
              "SourceMass": 1e6, "SourceRadiativeEfficiency": 0.1
             }                    
    
    # X-ray source, helium included         
    if pt == 5:
        pf = {"ProblemType": 5, "InterpolationMethod": 0, \
              "ColumnDensityBinsHI": 50, "ColumnDensityBinsHeI": 10, "ColumnDensityBinsHeII": 10, 
              "GridDimensions": 100, "LengthUnits": 1000 * cm_per_kpc, \
              "TimeUnits": s_per_myr, "CurrentTime": 0.0, "StopTime": 10.0, 
              "StartRadius": 0.01, "dtDataDump": 1, "DataDumpName": 'dd', \
              "Isothermal": 0, "MultiSpecies": 1, "SecondaryIonization": 1, "CosmologicalExpansion": 0, \
              "DensityProfile": 1, "InitialRedshift": 20, "TemperatureProfile": 0, "InitialTemperature": 1e2, \
              "IonizationProfile": 1, "InitialHIIFraction": 1e-4, "SourceType": 3, "SourceLifetime": 1e10, \
              "SourceMass": 1e6, "SourceRadiativeEfficiency": 0.1, "SpectrumPowerLawIndex": 1.5, \
              "SpectrumMinEnergy": 1e2, "SpectrumMaxEnergy": 1e4, "SpectrumMinNormEnergy": 1e2, "SpectrumMaxNormEnergy": 1e4 
             }      
             
    # X-ray source, helium included, discrete spectrum         
    if pt == 5.1:
        pf = {"ProblemType": 5.1, "InterpolationMethod": 0, \
              "ColumnDensityBinsHI": 50, "ColumnDensityBinsHeI": 10, "ColumnDensityBinsHeII": 10, 
              "GridDimensions": 100, "LengthUnits": 100 * cm_per_kpc, \
              "TimeUnits": s_per_myr, "CurrentTime": 0.0, "StopTime": 10.0, 
              "StartRadius": 0.01, "dtDataDump": 1, "DataDumpName": 'dd', \
              "Isothermal": 0, "MultiSpecies": 1, "SecondaryIonization": 1, "CosmologicalExpansion": 0, \
              "DensityProfile": 1, "InitialRedshift": 20, "TemperatureProfile": 0, "InitialTemperature": 1e2, \
              "IonizationProfile": 1, "InitialHIIFraction": 1e-4, "SourceType": 3, "SourceLifetime": 1e10, \
              "DiscreteSpectrumSED": [500.], "DiscreteSpectrumRelLum": [1.], "DiscreteSpectrum": 1,\
              "SourceMass": 1e6, "SourceRadiativeEfficiency": 0.1
             }                                 
             
        
    return pf    