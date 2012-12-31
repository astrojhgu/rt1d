"""

InitializeChemicalNetwork.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Sep 20 13:15:30 2012

Description: ChemicalNetwork object just needs to have methods called
'RateEquations' and 'Jacobian'

"""

import numpy as np
import dengo.primordial_rates, dengo.primordial_cooling
import dengo.oxygen_rates, dengo.oxygen_cooling
from dengo.chemistry_constants import tiny, kboltz, mh
from dengo.chemical_network import ChemicalNetwork, \
    reaction_registry, cooling_registry

class DengoChemicalNetwork:
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
            
            # NAMESPACE CONFLICT
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
            for ion in self.grid.ions_by_ion[element]:
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
        for species in self.grid.all_ions:
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
        
    def RateEquations(self, t, q, **kwargs):
        """
        Compute RHS of rate equations.  
        """
        
        dqdt = np.zeros_like(self.dqdt)
                        
        # Compute RHS of ODEs
        for i, ode in enumerate(self.dqdt):
            dqdt[i] = eval(self.dqdt[i]) 
                
        return dqdt
        
    def Jacobian(self, t, q, **kwargs):
        """
        Compute the Jacobian of the rate equations.
        """    

        jac = np.zeros([len(self.dqdt)] * 2)
        for i, sp1 in enumerate(self.grid.all_species):
            for j, sp2 in enumerate(self.grid.all_species):
                jac[i][j] = eval(self.jac[i][j])
                
        return jac   
        
class SimpleChemicalNetwork:
    def __init__(self, grid):
        self.grid = grid
        
        from ..physics.ComputeRateCoefficients import RateCoefficients
        self.coeff = RateCoefficients()

        self.Isothermal = self.grid.isothermal        
        self.MultiSpecies = (2 in self.grid.Z)

        # For convenience 
        self.zeros = np.zeros(len(self.grid.all_species))
        self.zeros_tmp = np.zeros([len(self.grid.all_species), self.grid.dims])
        self.zeros_jac = np.zeros([len(self.grid.all_species)] * 2)
        self.zeros_3xgrid = np.zeros([3, self.grid.dims])
        
    def RateEquations(self, t, q, args):
        """
        This function returns the right-hand side of our ODE's,
        Equations 1, 2, 3 and 9 in Mirocha et al. (2012), except
        we're solving for ion fractions instead of number densities.
        """       
         
        self.q = q    
            
        cell = args[0]
                
        n_H = self.grid.n_H[cell]
        
        xHI = q[0]
        xHII = q[1]
        
        # Neutrals (current time-step)
        nHI = n_H * xHI
        nHII = n_H * xHII
                
        if self.MultiSpecies:
            e = 5
            n_He = self.grid.element_abundances[1] * self.grid.n_H[cell]
            n_e = q[e]
            xHeI = q[2]
            xHeII = q[3]
            xHeIII = q[4]
            nHeI = n_He * xHeI
            nHeII = n_He * xHeII
            nHeIII = n_He * xHeIII
            
        else:
            e = 2
            n_e = q[e]
        
        # Source-dependent coefficients    
        # Gamma, gamma, k_H = args
        Gamma, k_H = [np.zeros(3)] * 2
        gamma = np.zeros([3, 3])
                                
        # Always solve hydrogen rate equation
        self.dqdt = self.zeros
        self.dqdt[0] = -(Gamma[0] + self.Beta[0][cell] * n_e) * xHI \
                     +   self.alpha[0][cell] * n_e * xHII
        self.dqdt[1] = -self.dqdt[0]   
        self.dqdt[e] = self.dqdt[1] * n_H             
                                
        # Helium rate equations  
        if self.MultiSpecies:
            self.dqdt[2] = -(Gamma[1] + self.Beta[1][cell] * n_e) * xHeI \
                         +  (self.alpha[1][cell] + self.xi[cell]) * n_e * xHeII
            self.dqdt[3] = (Gamma[1] + self.Beta[1][cell] * n_e) * xHeI \
                         +  self.alpha[2][cell] * n_e * xHeIII \
                         - (self.Beta[1][cell] + self.alpha[1][cell] \
                         +  self.xi[cell]) * n_e * xHeII \
                         -  Gamma[2] * xHeII
            self.dqdt[4] = (Gamma[2] + self.Beta[2][cell] * n_e) * xHeII \
                         - self.alpha[2][cell] * n_e * xHeIII
            self.dqdt[e] += (self.dqdt[3] + self.dqdt[4]) * n_He
                            
        # Temperature evolution - looks dumb but using np.sum is slow
        if not self.Isothermal:
            phoheat = k_H[0] * nHI
            ioncool = zeta[0] * nHI
            reccool = eta[0] * nHII
            exccool = psi[0] * nHI
            
            if self.MultiSpecies:
                phoheat += k_H[1] * nabs[1] + k_H[2] * nabs[2]
                ioncool += zeta[1] * nabs[1] + zeta[2] * nabs[2]
                reccool += eta[1] * nion[1] + eta[2] * nion[2]
                exccool += psi[1] * nabs[1] + psi[2] * nabs[2]
            
            self.dqdt[-1] = phoheat - n_e * (ioncool + reccool + exccool + nHeIII * omega[1])

        return self.dqdt
        
    def Jacobian(self, t, q, args):
        """
        Jacobian of the rate equations.
        """    
                    
        cell = args[0]            
                              
        Gamma, k_H = [np.zeros(3)] * 2
        gamma = np.zeros([3, 3])  
        
        n_H = self.grid.n_H[cell]
        
        xHI = q[0]
        xHII = q[1]                    
        nHI = n_H * xHI
        nHII = n_H * xHII
        
        if self.MultiSpecies:
            e = 5
            n_He = self.grid.element_abundances[1] * self.grid.n_H[cell]
            xHeI = q[2]
            xHeII = q[3]
            xHeIII = q[4]
            n_e = q[e]
            nHeI = n_He * xHeI
            nHeII = n_He * xHeII
            nHeIII = n_He * xHeIII 
        else:
            e = 2
            n_He = 0.0
            n_e = q[e]
            
        J = np.zeros_like(self.zeros_jac)

        # Hydrogen terms - diagonal
        J[0][0] = -(Gamma[0] + self.Beta[0][cell] * n_e) \
                -   self.alpha[0][cell] * n_e
        J[1][1] = J[0][0]
        
        # Hydrogen - off-diagonal
        J[0][1] = -J[0][0]
        J[1][0] = -J[0][0]
        
        # Electron elements
        J[0][e] = -self.Beta[0][cell] * xHI \
                +  self.alpha[0][cell] * xHII
        J[1][e] = -J[0][e]     
        J[e][0] = Gamma[0] * n_H + self.Beta[0][cell] * n_e * n_H \
                + self.alpha[0][cell] * n_e * n_H
        J[e][1] = -J[2][0]
        J[e][e] = self.Beta[0][cell] * n_H * xHI \
                - self.alpha[0][cell] * n_H * xHII
        
        if self.MultiSpecies:
            
            # First - diagonal elements
            J[2][2] = -(Gamma[1] + self.Beta[1][cell] * n_e) \
                    -  (self.alpha[1][cell] + self.xi[cell]) * n_e
            J[3][3] = -(Gamma[1] + self.Beta[0][cell] * n_e) \
                    -  (self.Beta[1][cell] + self.alpha[1][cell] \
                    +   self.xi[cell]) * n_e \
                    -   self.alpha[2][cell] * n_e \
                    -   Gamma[2]
            J[4][4] = -(Gamma[2] + self.Beta[2][cell] * n_e) \
                    -   self.alpha[2][cell] * n_e
            
            # Off-diagonal elements HeI
            J[2][3] = (Gamma[1] + self.Beta[0][cell] * n_e) \
                    + (self.alpha[1][cell] + self.xi[cell]) * n_e
            J[2][4] = (Gamma[1] + self.Beta[0][cell] * n_e) \
                    - (self.alpha[1][cell] + self.xi[cell]) * n_e 
            J[2][5] = -self.Beta[0][cell] * xHeI \
                    + (self.alpha[1][cell] + self.xi[cell]) * xHeII                
            
            # Off-diagonal elements HeII
            J[3][2] = (Gamma[1] + self.Beta[0][cell] * n_e) \
                    - (self.alpha[1][cell] + self.xi[cell]) * n_e \
                    + (self.Beta[1][cell] + self.alpha[1][cell] \
                    +  self.xi[cell]) * n_e
            J[3][4] = -J[3][2]
            J[3][5] = self.Beta[2][cell] * xHeIII \
                    - self.alpha[2][cell] * xHeIII
            
            # Off-diagonal elements HeIII
            J[4][2] = -(Gamma[1] + self.Beta[0][cell] * n_e) \
                    +   self.alpha[2][cell] * n_e
            J[4][3] = (Gamma[1] + self.Beta[0][cell] * n_e) \
                    +   self.alpha[2][cell] * n_e
            J[4][5] = self.Beta[2][cell] * xHeII \
                    - self.alpha[2][cell] * xHeIII            
            
            # Electrons
            J[e][e] += self.Beta[0][cell] * n_H * xHI \
                    + self.Beta[1][cell] * xHeI * n_He \
                    + self.Beta[2][cell] * xHeII * n_He \
                    - self.alpha[0][cell] * n_H * xHII \
                    - self.alpha[1][cell] * n_He * xHeII \
                    - self.alpha[2][cell] * n_He * xHeIII \
                    - self.xi[cell] * n_He * xHeII 
                    
            J[e][2] = 0
            J[e][3] = 0 
            J[e][4] = 0        
                    
            # H/He coupling terms
            
            # Gas energy                   
                                                               
        return J
                
    def SourceIndependentCoefficients(self, T):
        """
        Compute values of rate coefficients which depend only on 
        temperature and/or number densities of electrons/ions.
        """    
        
        self.T = T
        self.Beta = np.zeros_like(self.zeros_3xgrid)
        self.alpha = np.zeros_like(self.zeros_3xgrid)
        #self.zeta = np.zeros_like(self.zeros_3xgrid)
        #self.eta = np.zeros_like(self.zeros_3xgrid)
        #self.psi = np.zeros_like(self.zeros_3xgrid)
        self.xi = np.zeros(self.grid.dims)
        #self.omega = np.zeros_like(self.zeros_3xgrid)
        for i in xrange(3):
            if i > 0 and (not self.MultiSpecies):
                continue
                
            self.Beta[i,...] = self.coeff.CollisionalIonizationRate(i, T)
            self.alpha[i,...] = self.coeff.RadiativeRecombinationRate(i, T)
                        
        # Di-electric recombination
        self.xi = self.coeff.DielectricRecombinationRate(T)
                    