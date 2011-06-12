"""
InitializeParameterSpace.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-10-14.

Description: Because we allow parameters in the parameter file to be tuples or lists, this routine will 
determine all possible combinations of parameters and return a dictionary of dictionaries with the 
parameters for each individual run.  First, the method 'ReadParameterFile' simply parses the parameter
file we've supplied into a format the routine 'AllParameterSets' will understand.
     
"""

import numpy as np
import itertools as it
import copy
from SetDefaultParameterValues import *

cm_per_kpc = 3.08568 * 10**21
s_per_myr = 365.25 * 24 * 3600 * 10**6

class InitializeParameterSpace:
    def __init__(self, pf):
        self.pf = pf
        
    def ReadParameterFile(self):
        """
        Read in the parameter file, and parse the parameter names and arguments.
        Return a dictionary that contains all parameters and their values, whether 
        they be floats, tuples, or lists.
        """
        f = open(self.pf, "r")
        pf_dict = SetDefaultParameterValues()
        for line in f:
            if not line.split(): continue
            if line.split()[0][0] == "#": continue
            
            # Cleave off end-of-line comments.
            line = line[:line.rfind("#")].strip()
            
            # Read in the parameter name and the parameter value(s).
            parname, eq, parval = line.partition("=")
                        
            # ProblemType option
            if parname.strip() == 'ProblemType' and float(parval) > 0:
                pf_new = self.ProblemType(float(parval))
                for param in pf_new: pf_dict[param] = pf_new[param]
                
            # Else, actually read in the parameter                                     
            try: parval = float(parval)
            except ValueError:
                if parval.strip().isalnum(): 
                    parval = str(parval.strip())
                else:
                    parval = parval.strip().split(",")
                    tmp = []       
                    if parval[0][0] == '(':
                        for element in parval: 
                            if element.strip(" (,)").isdigit(): tmp.append(float(element.strip("(,)")))
                            else: tmp.append(element.strip(" (,)"))
                        parval = tuple(tmp)                    
                    elif parval[0][0] == '[':
                        for element in parval: tmp.append(float(element.strip("[,]")))
                        parval = list(tmp)
                    else:
                        raise ValueError('The format of this parameter is not understood.')
                    
            pf_dict[parname.strip()] = parval
                    
        return pf_dict
        
    def AllParameterSets(self):
        """
        Take the dictionary returned by 'ReadParameterFile' and construct a new dictionary,
        containing a dictionary for each individual parameter set.  If we're doing a single
        model run, this will be the same dictionary as that returned by 'ReadParameterFile'.
        """
        pf = self.ReadParameterFile()
        
        # First, construct a list of tuples, each one containing the name of the parameter 
        # and one possible value it will take on.
        tupct = 0
        allparvals = []
        for par in pf:
            if type(pf[par]) is tuple: 
                mi, ma, nbins, loglin = pf[par]
                mi = float(mi)
                ma = float(ma)
                nbins = float(nbins)
                if loglin == 'lin': parvals = np.arange(mi, ma * 1.0001, (ma - mi) / (nbins - 1))
                elif loglin == 'log': parvals = np.logspace(np.log10(mi), np.log10(ma), nbins)
                else: raise ValueError('Spacing must be linear or logarithmic.')
                                                                
                for val in parvals:
                    allparvals.append((par, val))
                
                tupct += 1
        
        pf_key = pf["SavePrefix"]
        
        # Now, construct a dictionary of dictionaries, one for each set of parameters.
        all_pfs_dict = {}  
        if tupct == 0: all_pfs_dict = {'{0}0000'.format(pf_key): pf}
        elif tupct == 1:
            allcombos = it.combinations(allparvals, tupct)
            
            for i, combo in enumerate(allcombos):
                tmp = copy.deepcopy(pf)
                tmp[combo[0][0]] = combo[0][1]
                all_pfs_dict["{0}{1:04d}".format(pf_key, i)] = copy.deepcopy(tmp) 
                del tmp
                                                    
        else:
            allcombos = it.combinations(allparvals, tupct)
            
            # First, eliminate parameter sets that have two of any parameter.
            goodcombos = []
            for combo in allcombos:
                tmp = []
                for element in combo: tmp.append(element[0])
                
                if len(set(tmp)) < len(tmp): continue
                else: goodcombos.append(combo)
                del tmp
                
            # Now that we know all the 'good' parameter combinations, make our dictionary.
            for i, combo in enumerate(goodcombos):
                tmp = copy.copy(pf)
                for par in combo:
                    tmp[par[0]] = par[1]
                                             
                all_pfs_dict["{0}{1:04d}".format(pf_key, i)] = tmp 
            
           
        return all_pfs_dict
        
    def ProblemType(self, pt):
        """
        Storage bin for predefined problem types, 'pt's, like those used in the radiative transfer comparison project ('RT'),
        or John and Tom's 2010 ENZO-MORAY ('EM') paper.
        """
        
        # EM-1, RT1: Pure hydrogen, isothermal HII region expansion
        if pt == 1:
            pf = {"ProblemType": 1, "IntegrationMethod": 0, "InterpolationMethod": 0, "MonitorSimulation": 0, \
                  "ColumnDensityBinsHI": 500, "ExitAfterIntegralTabulation": 0, "GridDimensions": 1000, "LengthUnits": 6.6 * cm_per_kpc, \
                  "TimeUnits": s_per_myr, "CurrentTime": 0.0, "StopTime": 500.0, "InitialTimestep": 0.1, "AdaptiveTimestep": 0,  \
                  "StartRadius": 0.001, "MaxHIIFraction": 0.9999, "dtDataDump": 5.0, "DataDumpName": 'dd', \
                  "SavePrefix": 'rt', "SolveTemperatureEvolution": 0, "MultiSpecies": 0, "SecondaryElectronMethod": 0, "CosmologicalExpansion": 0, \
                  "DensityProfile": 0, "InitialDensity": 1e-3, "TemperatureProfile": 0, "InitialTemperature": 1e4, \
                  "IonizationProfile": 1, "InitialHIIFraction": 1.2e-3, "SourceType": 0, "SourceLifetime": 500.0, \
                  "SpectrumPhotonLuminosity": 5e48, "DiscreteSpectrumMethod": 1, "DiscreteSpectrumSED": [13.61], \
                  "SpectrumMinEnergy": 0.1, "SpectrumMaxEnergy": 100
                 }    
                 
        # TZ08, RT1, EM-1: Pure hydrogen, isothermal (with parameters of TZ07) not sure about initial HIIFraction  
        if pt == 1.1:
            pf = {"ProblemType": 1.1, "IntegrationMethod": 0, "InterpolationMethod": 0, "MonitorSimulation": 0, \
                  "ColumnDensityBinsHI": 500, "ExitAfterIntegralTabulation": 0, "GridDimensions": 1000, "LengthUnits": 100. * cm_per_kpc, \
                  "TimeUnits": s_per_myr, "CurrentTime": 0.0, "StopTime": 100.0, "InitialTimestep": 0.1, "AdaptiveTimestep": 0,  \
                  "StartRadius": 0.001, "MaxHIIFraction": 0.9999, "dtDataDump": 1.0, "DataDumpName": 'dd', \
                  "SavePrefix": 'rt', "SolveTemperatureEvolution": 0, "MultiSpecies": 0, "SecondaryElectronMethod": 0, "CosmologicalExpansion": 0, \
                  "DensityProfile": 0, "InitialDensity": 1.87e-4, "TemperatureProfile": 0, "InitialTemperature": 1e4, \
                  "IonizationProfile": 1, "InitialHIIFraction": 1e-4, "SourceType": 0, "SourceLifetime": 500.0, \
                  "SpectrumPhotonLuminosity": 1e54, "DiscreteSpectrumMethod": 1, "DiscreteSpectrumSED": [13.6], \
                  "SpectrumMinEnergy": 0.1, "SpectrumMaxEnergy": 100
                 }           
        
        # EM-2: Pure hydrogen, HII region expansion, temperature evolution allowed, 4-bin spectrum (supposedly samples 1e5 K BB)
        if pt == 2:
            pf = {"ProblemType": 2, "UseScipy": 1, "IntegrationMethod": 0, "InterpolationMethod": 0, "MonitorSimulation": 0, \
                  "ColumnDensityBinsHI": 500, "ExitAfterIntegralTabulation": 0, "GridDimensions": 1000, "LengthUnits": 6.6 * cm_per_kpc, \
                  "TimeUnits": s_per_myr, "CurrentTime": 0.0, "StopTime": 500.0, "InitialTimestep": 0.1, "AdaptiveTimestep": 0,  \
                  "StartRadius": 0.001, "MaxHIIFraction": 0.9999, "dtDataDump": 10., "DataDumpName": 'dd', \
                  "SavePrefix": 'rt', "SolveTemperatureEvolution": 1, "MultiSpecies": 0, "SecondaryElectronMethod": 0, "CosmologicalExpansion": 0, \
                  "DensityProfile": 0, "InitialDensity": 1e-3, "TemperatureProfile": 0, "InitialTemperature": 1e2, \
                  "IonizationProfile": 1, "InitialHIIFraction": 1.2e-3, "SourceType": 1, "SourceLifetime": 500.0, \
                  "SourceTemperature": 1e5, "DiscreteSpectrumMethod": 1, "DiscreteSpectrumSED": [16.74, 24.65, 34.49, 52.06], \
                  "DiscreteSpectrumRelLum": [0.277, 0.335, 0.2, 0.188], "SpectrumMinEnergy": 0.1, "SpectrumMaxEnergy": 100
                 }      
                
        # EM-2: Everything the same, except for complete (non-discretized) BB spectrum         
        if pt == 2.1:
            pf = {"ProblemType": 2.1, "UseScipy": 1, "IntegrationMethod": 0, "InterpolationMethod": 0, "MonitorSimulation": 0, \
                  "ColumnDensityBinsHI": 500, "ExitAfterIntegralTabulation": 0, "GridDimensions": 1000, "LengthUnits": 6.6 * cm_per_kpc, \
                  "TimeUnits": s_per_myr, "CurrentTime": 0.0, "StopTime": 500.0, "InitialTimestep": 0.1, "AdaptiveTimestep": 0,  \
                  "StartRadius": 0.001, "MaxHIIFraction": 0.9999, "dtDataDump": 10., "DataDumpName": 'dd', \
                  "SavePrefix": 'rt', "SolveTemperatureEvolution": 1, "MultiSpecies": 0, "SecondaryElectronMethod": 0, "CosmologicalExpansion": 0, \
                  "DensityProfile": 0, "InitialDensity": 1e-3, "TemperatureProfile": 0, "InitialTemperature": 1e2, \
                  "IonizationProfile": 1, "InitialHIIFraction": 1.2e-3, "SourceType": 1, "SourceLifetime": 500.0, \
                  "SourceTemperature": 1e5, "DiscreteSpectrumMethod": 0, "SpectrumMinEnergy": 0.1, "SpectrumMaxEnergy": 100,
                 }               
                
            
        return pf    
            
            
            
            
            
            
            
        
        
