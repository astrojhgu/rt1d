"""

InitializeChemicalNetwork.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Sep 20 13:15:30 2012

Description: 

"""

import numpy as np

# Namespace conflict!
from dengo.chemical_network import \
    ChemicalNetwork, \
    reaction_registry, \
    cooling_registry
import dengo.primordial_rates, dengo.primordial_cooling
import dengo.oxygen_rates, dengo.oxygen_cooling
from dengo.chemistry_constants import tiny, kboltz, mh

class ChemicalNetworkThing:
    def __init__(self, grid):
        self.grid = grid
        
        # Make list of dengo.chemical_network.ChemicalNetwork objects
        self._initialize()
    
    def _initialize(self):
        """
        Create list of ChemicalNetwork objects (one for each element).
        """
        
        self.networks = {}
        for element in self.grid.elements:
            
            chem_net = ChemicalNetwork()
            
            # Add cooling actions
            for ca in cooling_registry.values():
                if ca.name.startswith("%s_" % element):
                    chem_net.add_cooling(ca)
                    
            # Add ionization reactions        
            for s in reaction_registry.values():
                if s.name.startswith("%s_" % element):
                    chem_net.add_reaction(s)              
                    
            self.networks[element] = chem_net
        
        self._set_RHS()
        self._set_Jac()                 
            
    def _set_RHS(self):
        """
        Create string representations of RHS of all ODEs.
        
        NOTE: This will not work with primordial network naming conventions.
        """    
        
        # Set up RHS of rate equations - must convert dengo strings to ours  
        i = 0
        dedot = ''
        gedot = ''
        self.dqdt = []
        for element in self.grid.elements:
            for ion in self.grid.ions[element]:
                expr = self.networks[element].rhs_string_equation(ion)
                expr = self._translate_rate_str(expr)                
                self.dqdt.append(expr)        
                i += 1
                
            de = self._translate_rate_str(self.networks[element].rhs_string_equation('de'))   
            dedot += '+%s' % de  

            if not self.grid.isothermal:
                ge = self._translate_cool_str(self.networks[element].cooling_string_equation())               
                gedot += '+%s' % ge  
        
        # Electrons and gas energy
        self.dqdt.append(dedot)
        if not self.grid.isothermal:
            self.dqdt.append(gedot)
            
    def _set_Jac(self):
        """
        Create string representations of Jacobian.
        """        
        
        self.jac = []
        
        for element in self.networks:
            chemnet = self.networks[element]
            for i, sp1 in enumerate(self.grid.all_species):
                self.jac.append([])
                for j, sp2 in enumerate(self.grid.all_species):
                    expr = chemnet.jacobian_string_equation(sp1, sp2)
                    expr = self._translate_rate_str(expr)
                    self.jac[i].append(expr)                
            
    def _translate_rate_str(self, expr):
        """
        Convert dengo style rate equation strings to our dictionary / vector convention.
        """    
        
        # Ionization and recombination coefficients
        for species in self.grid.ion_species:
            expr = expr.replace('%s_i' % species, 'kwargs[\'%s_i\']' % species)
            expr = expr.replace('%s_r' % species, 'kwargs[\'%s_r\']' % species)
        
        # Ions, de, or ge
        for species in self.grid.all_species:    
            expr = expr.replace('*%s' % species, '*q[%i]' % list(self.grid.qmap).index(species))  
              
        # Cooling terms      
        expr = self._translate_cool_str(expr)      
              
        return expr      
        
    def _translate_cool_str(self, expr):
        """
        Convert dengo style energy equation strings to dictionary / vector convention
        """      
        
        return expr
        
    def RateEquations(self, t, q, kwargs):
        """
        Compute RHS of rate equations.  
        """
        
        dqdt = np.zeros_like(self.dqdt)
                        
        # Compute RHS of ODEs
        for i, ode in enumerate(self.dqdt):
            dqdt[i] = eval(self.dqdt[i]) 
                
        return dqdt
        
    def Jacobian(self, t, q, kwargs):
        """
        Compute the Jacobian of the rate equations.
        """    

        jac = np.zeros([len(self.dqdt)] * 2)
        for i, sp1 in enumerate(self.grid.all_species):
            for j, sp2 in enumerate(self.grid.all_species):
                jac[i][j] = eval(self.jac[i][j])
                
        return jac        
        
        
        
        
        
