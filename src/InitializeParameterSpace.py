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
                        
            try: parval = float(parval)
            except ValueError:
                if parval.strip().isalnum(): 
                    parval = str(parval)
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
        containing a dictionary for each individual parameter set.  IF we're doing a single
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
                if loglin == 'lin': parvals = np.arange(mi, ma * 1.0001, (ma - mi) / (nbins - 1))
                elif loglin == 'log': parvals = np.logspace(np.log10(mi), np.log10(ma), nbins)
                else: raise ValueError('Spacing must be linear or logarithmic.')
                
                for val in parvals:
                    allparvals.append((par, val))
                
                tupct += 1
        
        pf_key = 'psp'      # make this a parameter
        
        # Now, construct a dictionary of dictionaries, one for each set of parameters.
        all_pfs_dict = {}  
        if tupct == 0: all_pfs_dict = {'{0}1'.format(pf_key): pf}
        elif tupct == 1:
            allcombos = it.combinations(allparvals, tupct)
            
            tmp = pf
            for i, combo in enumerate(allcombos):
                tmp[combo[0][0]] = combo[0][1]
                all_pfs_dict["{0}{1:05d}".format(pf_key, i)] = tmp
            
            return all_pfs_dict
            
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
                                             
                all_pfs_dict["{0}{1:05d}".format(pf_key, i)] = tmp
            
        return all_pfs_dict
        