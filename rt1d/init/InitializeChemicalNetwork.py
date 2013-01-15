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

        # For convenience 
        self.zeros_q = np.zeros(len(self.grid.all_species))
        self.zeros_jac = np.zeros([len(self.grid.all_species)] * 2)
        self.zeros_3xgrid = np.zeros([3, self.grid.dims])
        
    def RateEquations(self, t, q, args):
        """
        This function returns the right-hand side of our ODE's,
        Equations 1, 2, 3 and 9 in Mirocha et al. (2012), except
        we're solving for ion fractions instead of number densities.
        """       
         
        self.q = q
        self.dqdt = np.zeros_like(self.zeros_q)
        
        cell, Gamma, gamma, k_H = args
                    
        n_H = self.grid.n_H[cell]
        
        if 1 in self.grid.Z:
            h1, h2, e = (0, 1, 2)
            xHI = q[h1]
            xHII = q[h2]        
            nHI = n_H * xHI
            nHII = n_H * xHII
                
            if 2 in self.grid.Z:
                he1, he2, he3, e = (2, 3, 4, 5)
                n_He = self.grid.element_abundances[1] * self.grid.n_H[cell]
                xHeI = q[he1]
                xHeII = q[he2]
                xHeIII = q[he3]
                
        elif 2 in self.grid.Z:    
            he1, he2, he3, e = (0, 1, 2, 3)
            n_He = self.grid.element_abundances[0] * self.grid.n_H[cell]
            xHeI = q[he1]
            xHeII = q[he2]
            xHeIII = q[he3]
        
        n_e = q[e]
                                            
        # Hydrogen rate equations
        if 1 in self.grid.Z:
            self.dqdt[h1] = -(Gamma[0] + self.Beta[cell][0] * n_e) * xHI \
                          +   self.alpha[cell][0] * n_e * xHII \
                          -   gamma[0][0] * xHI # plus gamma[0][1:]
            self.dqdt[h2] = -self.dqdt[0]   
            self.dqdt[e] = self.dqdt[h2] * n_H             
                                
        # Helium rate equations  
        if 2 in self.grid.Z:
            self.dqdt[he1] = -(Gamma[1] + self.Beta[cell][1] * n_e) * xHeI \
                           +  (self.alpha[cell][1] + self.xi[cell]) * n_e * xHeII
            self.dqdt[he2] = (Gamma[1] + self.Beta[cell][1] * n_e) * xHeI \
                           +  self.alpha[cell][2] * n_e * xHeIII \
                           - (self.Beta[cell][1] + self.alpha[cell][1] \
                           +  self.xi[cell]) * n_e * xHeII \
                           -  Gamma[2] * xHeII
            self.dqdt[he3] = (Gamma[2] + self.Beta[cell][2] * n_e) * xHeII \
                           - self.alpha[cell][2] * n_e * xHeIII
            self.dqdt[e] += (self.dqdt[he2] + self.dqdt[he3]) * n_He
                            
        # Temperature evolution - looks dumb but using np.sum is slow
        if not self.Isothermal:
            phoheat = 0.0
            ioncool = 0.0
            reccool = 0.0
            exccool = 0.0
            
            if 1 in self.grid.Z:
                phoheat += k_H[h1] * xHI * n_H
                ioncool += self.zeta[cell][h1] * xHI * n_H
                reccool += self.eta[cell][h1] * xHII * n_H
                exccool += self.psi[cell][h1] * xHI * n_H
            
            #if 2 in self.grid.Z:
            #    phoheat += k_H[1] * nabs[1] + k_H[2] * nabs[2]
            #    ioncool += zeta[1] * nabs[1] + zeta[2] * nabs[2]
            #    reccool += eta[1] * nion[1] + eta[2] * nion[2]
            #    exccool += psi[1] * nabs[1] + psi[2] * nabs[2]
            
            self.dqdt[-1] = phoheat - n_e * (ioncool + reccool + exccool)

            # Multispecies : dqdt[-1] += n_e * xHeIII * n_He * omega

        return self.dqdt
        
    def Jacobian(self, t, q, args):
        """
        Jacobian of the rate equations.
        """    
                    
        cell, Gamma, gamma, k_H = args       
        
        n_H = self.grid.n_H[cell]
        
        if 1 in self.grid.Z:
            h1, h2, e = (0, 1, 2)
            xHI = q[h1]
            xHII = q[h2]        
            nHI = n_H * xHI
            nHII = n_H * xHII
            if 2 in self.grid.Z:
                he1, he2, he3, e = (2, 3, 4, 5)
                n_He = self.grid.element_abundances[1] * self.grid.n_H[cell]
                xHeI = q[he1]
                xHeII = q[he2]
                xHeIII = q[he3]
                nHeI = n_He * xHeI
                nHeII = n_He * xHeII
                nHeIII = n_He * xHeIII
        elif 2 in self.grid.Z:    
            he1, he2, he3, e = (0, 1, 2, 3)
            n_He = self.grid.element_abundances[0] * self.grid.n_H[cell]
            xHeI = q[he1]
            xHeII = q[he2]
            xHeIII = q[he3]
            nHeI = n_He * xHeI
            nHeII = n_He * xHeII
            nHeIII = n_He * xHeIII
        
        ge = q[-1]
        n_e = q[e]
            
        J = np.zeros_like(self.zeros_jac)

        # Hydrogen terms - diagonal
        if 1 in self.grid.Z:
            J[h1][h1] = -(Gamma[0] + self.Beta[cell][0] * n_e) \
                      -   self.alpha[cell][0] * n_e \
                      -   gamma[0][0]
            J[h2][h2] = J[h1][h1]
            
            # Hydrogen - off-diagonal
            J[h1][h2] = -J[h1][h1]
            J[h1][e] = -self.Beta[cell][0] * xHI \
                     +  self.alpha[cell][0] * xHII
            J[h2][h1] = -J[h1][h1]
            J[h2][e] = -J[h1][e]
            
            # Electron elements
            J[e][h1] = Gamma[0] * n_H + self.Beta[cell][0] * n_e * n_H \
                     + self.alpha[cell][0] * n_e * n_H \
                     + np.sum(gamma[0]) * n_H
            J[e][h2] = -J[e][h2]
            J[e][e] = self.Beta[cell][0] * n_H * xHI \
                    - self.alpha[cell][0] * n_H * xHII
                    
            # Gas energy
            J[-1][h1] = n_H * (k_H[h1] \
                      - n_e * (self.zeta[cell][h1] + self.psi[cell][h1]
                      - self.eta[cell][h1]))
            J[-1][h2] = -J[-1][h1]
            J[-1][-1] = 0

        if 2 in self.grid.Z:
            
            # First - diagonal elements
            J[he1][he1] = -(Gamma[1] + self.Beta[cell][1] * n_e) \
                        -  (self.alpha[cell][1] + self.xi[cell]) * n_e
            J[he2][he2] = -(Gamma[1] + self.Beta[cell][0] * n_e) \
                        -  (self.Beta[cell][1] + self.alpha[cell][1] \
                        +   self.xi[cell]) * n_e \
                        -   self.alpha[cell][2] * n_e \
                        -   Gamma[2]
            J[he3][he3] = -(Gamma[2] + self.Beta[cell][2] * n_e) \
                        -   self.alpha[cell][2] * n_e
            
            # Off-diagonal elements HeI
            J[he1][he2] = (Gamma[1] + self.Beta[cell][0] * n_e) \
                        + (self.alpha[cell][1] + self.xi[cell]) * n_e
            J[he1][he3] = (Gamma[1] + self.Beta[cell][0] * n_e) \
                        - (self.alpha[cell][1] + self.xi[cell]) * n_e 
            J[he1][e] = -self.Beta[cell][0] * xHeI \
                      + (self.alpha[cell][1] + self.xi[cell]) * xHeII                
            
            # Off-diagonal elements HeII
            J[he2][he1] = (Gamma[1] + self.Beta[cell][0] * n_e) \
                        - (self.alpha[cell][1] + self.xi[cell]) * n_e \
                        + (self.Beta[cell][1] + self.alpha[cell][1] \
                        +  self.xi[cell]) * n_e
            J[he2][he3] = -J[3][2]
            J[he2][e] = self.Beta[cell][2] * xHeIII \
                      - self.alpha[cell][2] * xHeIII
            
            # Off-diagonal elements HeIII
            J[he3][he1] = -(Gamma[1] + self.Beta[cell][0] * n_e) \
                        +   self.alpha[cell][2] * n_e
            J[he3][he2] = (Gamma[1] + self.Beta[cell][0] * n_e) \
                        +   self.alpha[cell][2] * n_e
            J[he3][e] = self.Beta[cell][2] * xHeII \
                      - self.alpha[cell][2] * xHeIII            
            
            # Electrons
            J[e][e] += self.Beta[cell][1] * xHeI \
                    + self.Beta[cell][2] * xHeII \
                    - self.alpha[cell][1] * xHeII \
                    - self.alpha[cell][2] * xHeIII \
                    - self.xi[cell] * xHeII 
                    
            #J[e][he1] = Gamma[1] - Gamma[2] \
            #          + (self.Beta[1][cell] - self.Beta[2][cell]) * n_e \
            #          + self.xi[cell] * n_e \
            #          + self.alpha[1][cell] * n_e + self.alpha[2][cell] * n_e 
            #J[e][he2] = Gamma[2] - Gamma[1] \
            #          + (self.Beta[2][cell] - self.Beta[1][cell]) * n_e \
            #          - self.xi[cell] * n_e \
            #          - self.alpha[1][cell] * n_e + self.alpha[2][cell] * n_e
            #J[e][he3] = -(Gamma[1] + Gamma[2]) \
            #          - (self.Beta[1][cell] + self.Beta[2][cell]) * n_e \
            #          + self.xi[cell] * n_e \
            #          + self.alpha[1][cell] * n_e - self.alpha[2][cell] * n_e
            
            J[e][e] *= n_He
                                
            # Gas energy
            
                                                   
        return J
                
    def SourceIndependentCoefficients(self, T):
        """
        Compute values of rate coefficients which depend only on 
        temperature and/or number densities of electrons/ions.
        """    
        
        self.T = T
        self.Beta = np.zeros_like(self.grid.zeros_grid_x_absorbers)
        self.alpha = np.zeros_like(self.grid.zeros_grid_x_absorbers)
        self.zeta = np.zeros_like(self.grid.zeros_grid_x_absorbers)
        self.eta = np.zeros_like(self.grid.zeros_grid_x_absorbers)
        self.psi = np.zeros_like(self.grid.zeros_grid_x_absorbers)
        
        for i in xrange(3):
            if (i == 0) and (1 not in self.grid.Z):
                continue
            
            if i > 0 and (2 not in self.grid.Z):
                continue
                
            self.Beta[...,i] = self.coeff.CollisionalIonizationRate(i, T)
            self.alpha[...,i] = self.coeff.RadiativeRecombinationRate(i, T)
            self.zeta[...,i] = self.coeff.CollisionalIonizationCoolingRate(i, T)
            self.eta[...,i] = self.coeff.RecombinationCoolingRate(i, T)
            self.psi[...,i] = self.coeff.CollisionalExcitationCoolingRate(i, T)
                        
        # Di-electric recombination
        self.xi = self.coeff.DielectricRecombinationRate(T)
        self.omega = self.coeff.DielectricRecombinationCoolingRate(T)
        
        return {'Beta': self.Beta, 'alpha': self.alpha,  
                'zeta': self.zeta, 'eta': self.eta, 'psi': self.psi, 
                'xi': self.xi, 'omega': self.omega}
        
                    