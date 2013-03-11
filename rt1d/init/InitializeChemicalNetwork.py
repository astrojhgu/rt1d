"""

InitializeChemicalNetwork.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Sep 20 13:15:30 2012

Description: ChemicalNetwork object just needs to have methods called
'RateEquations' and 'Jacobian'

"""

import numpy as np
from ..physics.Constants import k_B

try:
    import dengo.primordial_rates, dengo.primordial_cooling
    import dengo.oxygen_rates, dengo.oxygen_cooling
    from dengo.chemical_network import ChemicalNetwork, \
        reaction_registry, cooling_registry
except ImportError:
    pass   

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

        self.isothermal = self.grid.isothermal        

        # For convenience 
        self.zeros_q = np.zeros(len(self.grid.all_species))
        self.zeros_jac = np.zeros([len(self.grid.all_species)] * 2)
        
    def RateEquations(self, t, q, args):
        """
        This function returns the right-hand side of our ODE's,
        Equations 1, 2, 3 and 9 in Mirocha et al. (2012), except
        we're solving for ion fractions instead of number densities.
        """       
         
        self.q = q
        self.dqdt = np.zeros_like(self.zeros_q)
        
        cell, Gamma, gamma, k_H, n = args
                                    
        n_H = self.grid.n_H[cell]
        
        # h1, he1, etc. correspond to indices in absorbers list.
        
        if 1 in self.grid.Z:
            h1, e = (0, 2)
            xHI = q[h1]
            xHII = q[h1 + 1]        
            nHI = n_H * xHI
            nHII = n_H * xHII
                
            if 2 in self.grid.Z:
                he1, he2, e = (1, 2, 4)
                n_He = self.grid.element_abundances[1] * self.grid.n_H[cell]
                xHeI = q[he1]
                xHeII = q[he2]
                xHeIII = q[he2 + 1]
                
        elif 2 in self.grid.Z:    
            he1, he2, e = (0, 1, 3)
            n_He = self.grid.element_abundances[0] * self.grid.n_H[cell]
            xHeI = q[he1]
            xHeII = q[he2]
            xHeIII = q[he2 + 1]
        
        n_e = q[e]
                                                    
        # Hydrogen rate equations
        if 1 in self.grid.Z:
            self.dqdt[h1] = -(Gamma[h1] + self.Beta[cell][h1] * n_e) * xHI \
                          +   self.alpha[cell][h1] * n_e * xHII \
                          -   gamma[h1][h1] * xHI # plus gamma[0][1:]
            self.dqdt[h1 + 1] = -self.dqdt[h1]   
            self.dqdt[e] = self.dqdt[h1 + 1] * n_H             
                                
        # Helium rate equations  
        if 2 in self.grid.Z:
            
            self.dqdt[he1] = -(Gamma[he1] + self.Beta[cell][he1] * n_e) * xHeI \
                           +  (self.alpha[cell][he1] + self.xi[cell]) * n_e * xHeII
            self.dqdt[he2] = (Gamma[he1] + self.Beta[cell][he1] * n_e) * xHeI \
                           +  self.alpha[cell][he2] * n_e * xHeIII \
                           - (self.Beta[cell][he1] + self.alpha[cell][he1] \
                           +  self.xi[cell]) * n_e * xHeII \
                           -  Gamma[he2] * xHeII
            self.dqdt[he2 + 1] = (Gamma[he2] + self.Beta[cell][he2] * n_e) * xHeII \
                           - self.alpha[cell][he2] * n_e * xHeIII
            self.dqdt[e] += (self.dqdt[he2] + self.dqdt[he2 + 1]) * n_He
                            
        # Temperature evolution
        if not self.isothermal:
            phoheat, ioncool, reccool, exccool = np.zeros(4)
            
            if 1 in self.grid.Z:
                phoheat += k_H[h1] * xHI * n_H
                ioncool += self.zeta[cell][h1] * xHI * n_H
                reccool += self.eta[cell][h1] * xHII * n_H
                exccool += self.psi[cell][h1] * xHI * n_H
            
            if 2 in self.grid.Z:
                phoheat += k_H[he1] * xHeI * n_He + k_H[he2] * xHeII * n_He
                ioncool += self.zeta[cell][he1] * xHeI * n_He \
                         + self.zeta[cell][he2] * xHeI * n_He
                reccool += self.eta[cell][he1] * xHeII * n_He \
                         + self.eta[cell][he2] * xHeIII * n_He
                exccool += self.psi[cell][he1] * xHeI * n_He \
                         + self.psi[cell][he2] * xHeII * n_He \
            
            self.dqdt[-1] = phoheat - n_e * (ioncool + reccool + exccool)

            # Multispecies : dqdt[-1] += n_e * xHeIII * n_He * omega

        return self.dqdt
        
    def Jacobian(self, t, q, args):
        """
        Jacobian of the rate equations.
        """    
                    
        cell, Gamma, gamma, k_H, n = args
                
        n_H = self.grid.n_H[cell]
        
        dTde = 2. / 3. / k_B / n
            
        if 1 in self.grid.Z:
            h1, e = (0, 2)
            xHI = q[h1]
            xHII = q[h1 + 1]        
            nHI = n_H * xHI
            nHII = n_H * xHII
                
            if 2 in self.grid.Z:
                he1, he2, e = (1, 2, 4)
                n_He = self.grid.element_abundances[1] * self.grid.n_H[cell]
                xHeI = q[he1]
                xHeII = q[he2]
                xHeIII = q[he2 + 1]
                nHeI = n_He * xHeI
                nHeII = n_He * xHeII
                nHeIII = n_He * xHeIII
        elif 2 in self.grid.Z:    
            he1, he2, e = (0, 1, 3)
            n_He = self.grid.element_abundances[0] * self.grid.n_H[cell]
            xHeI = q[he1]
            xHeII = q[he2]
            xHeIII = q[he2 + 1]
            nHeI = n_He * xHeI
            nHeII = n_He * xHeII
            nHeIII = n_He * xHeIII            
        
        ge = q[-1]
        n_e = q[e]
            
        J = np.zeros_like(self.zeros_jac)

        # Hydrogen terms - diagonal
        if 1 in self.grid.Z:
            J[h1][h1] = -(Gamma[h1] + self.Beta[cell][h1] * n_e) \
                      -   self.alpha[cell][h1] * n_e \
                      -   gamma[h1][h1]
            J[h1 + 1][h1 + 1] = J[h1][h1]
            
            # Hydrogen - off-diagonal
            J[h1][h1 + 1] = -J[h1][h1]
            J[h1][e] = -self.Beta[cell][h1] * xHI \
                     +  self.alpha[cell][h1] * xHII
            J[h1 + 1][h1] = -J[h1][h1]
            J[h1 + 1][e] = -J[h1][e]
            
            # Electron elements
            J[e][h1] = Gamma[h1] * n_H + self.Beta[cell][h1] * n_e * n_H \
                     + self.alpha[cell][h1] * n_e * n_H \
                     + np.sum(gamma[h1]) * n_H
            J[e][h1 + 1] = -J[e][h1 + 1]
            J[e][e] = self.Beta[cell][h1] * n_H * xHI \
                    - self.alpha[cell][h1] * n_H * xHII
                    
            # Gas energy
            if not self.isothermal:
                #J[h1][-1] = 
                
                
                J[-1][h1] = n_H * (k_H[h1] \
                          - n_e * (self.zeta[cell][h1] 
                          + self.psi[cell][h1] - self.eta[cell][h1]))
                J[-1][h1 + 1] = -J[-1][h1]
                J[-1][-1] = 0

        if 2 in self.grid.Z:
            
            # First - diagonal elements
            J[he1][he1] = -(Gamma[he1] + self.Beta[cell][he1] * n_e) \
                        -  (self.alpha[cell][he1] + self.xi[cell]) * n_e
            J[he2][he2] = -(Gamma[he1] + self.Beta[cell][he1] * n_e) \
                        -  (self.Beta[cell][he1] + self.alpha[cell][he1] \
                        +   self.xi[cell]) * n_e \
                        -   self.alpha[cell][he2] * n_e \
                        -   Gamma[he2]
            J[he2 + 1][he2 + 1] = -(Gamma[he2] + self.Beta[cell][he2] * n_e) \
                                -   self.alpha[cell][he2] * n_e
            
            # Off-diagonal elements HeI
            J[he1][he2] = (Gamma[he1] + self.Beta[cell][he1] * n_e) \
                        + (self.alpha[cell][1] + self.xi[cell]) * n_e
            J[he1][he2 + 1] = (Gamma[he1] + self.Beta[cell][he1] * n_e) \
                            - (self.alpha[cell][he1] + self.xi[cell]) * n_e 
            J[he1][e] = -self.Beta[cell][he1] * xHeI \
                      + (self.alpha[cell][he1] + self.xi[cell]) * xHeII                
            
            # Off-diagonal elements HeII
            J[he2][he1] = (Gamma[he1] + self.Beta[cell][he1] * n_e) \
                        - (self.alpha[cell][he1] + self.xi[cell]) * n_e \
                        + (self.Beta[cell][he1] + self.alpha[cell][he1] \
                        +  self.xi[cell]) * n_e
            J[he2][he2 + 1] = -J[he2 + 1][he2]
            J[he2][e] = self.Beta[cell][he2] * xHeIII \
                      - self.alpha[cell][he2] * xHeIII
            
            # Off-diagonal elements HeIII
            J[he2 + 1][he1] = -(Gamma[he1] + self.Beta[cell][0] * n_e) \
                            +   self.alpha[cell][he2] * n_e
            J[he2 + 1][he2] = (Gamma[he1] + self.Beta[cell][0] * n_e) \
                            +   self.alpha[cell][he2] * n_e
            J[he2 + 1][e] = self.Beta[cell][he2] * xHeII \
                            - self.alpha[cell][he2] * xHeIII            
            
            # Electrons
            J[e][e] += self.Beta[cell][he1] * xHeI \
                    + self.Beta[cell][he2] * xHeII \
                    - self.alpha[cell][he1] * xHeII \
                    - self.alpha[cell][he2] * xHeIII \
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
        
        self.xi = np.zeros(self.grid.dims)
        self.omega = np.zeros(self.grid.dims)
        
        for i, absorber in enumerate(self.grid.absorbers):
            self.Beta[...,i] = self.coeff.CollisionalIonizationRate(i, T)
            self.alpha[...,i] = self.coeff.RadiativeRecombinationRate(i, T)
            self.zeta[...,i] = self.coeff.CollisionalIonizationCoolingRate(i, T)
            self.eta[...,i] = self.coeff.RecombinationCoolingRate(i, T)
            self.psi[...,i] = self.coeff.CollisionalExcitationCoolingRate(i, T)            
                        
        # Di-electric recombination
        if 2 in self.grid.Z:
            self.xi = self.coeff.DielectricRecombinationRate(T)
            self.omega = self.coeff.DielectricRecombinationCoolingRate(T)
        
        return {'Beta': self.Beta, 'alpha': self.alpha,  
                'zeta': self.zeta, 'eta': self.eta, 'psi': self.psi, 
                'xi': self.xi, 'omega': self.omega}
        
                    