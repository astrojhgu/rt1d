"""

ChemicalNetwork.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Sep 20 13:15:30 2012

Description: ChemicalNetwork object just needs to have methods called
'RateEquations' and 'Jacobian'

"""

import copy, sys
import numpy as np
from ..util import convert_ion_name
from ..physics.Constants import k_B, sigma_T, m_e, c, s_per_myr, erg_per_ev, h

try:
    have_dengo = True
    import dengo.primordial_rates, dengo.primordial_cooling
    import dengo.oxygen_rates, dengo.oxygen_cooling
    import dengo.carbon_rates, dengo.carbon_cooling
    import dengo.magnesium_rates, dengo.magnesium_cooling
    import dengo.neon_rates, dengo.neon_cooling
    import dengo.nitrogen_rates, dengo.nitrogen_cooling
    import dengo.silicon_rates, dengo.silicon_cooling
    import dengo.sulfur_rates, dengo.sulfur_cooling
    from dengo.chemical_network import ChemicalNetwork, \
        reaction_registry, cooling_registry
except ImportError:
    have_dengo = False 
        
class SimpleChemicalNetwork:
    def __init__(self, grid, rate_src='fk94'):
        self.grid = grid
        
        from ..physics.RateCoefficients import RateCoefficients
        self.coeff = RateCoefficients(grid, rate_src=rate_src)

        self.isothermal = self.grid.isothermal
        self.secondary_ionization = self.grid.secondary_ionization
        
        # For convenience 
        self.zeros_q = np.zeros(len(self.grid.evolving_fields))
        self.zeros_jac = np.zeros([len(self.grid.evolving_fields)] * 2)
        
    def _parse_q(self, q, n_H):
        x, n = {}, {}
        if 1 in self.grid.Z:
            x['h_1'] = q[self.grid.qmap.index('h_1')]
            x['h_2'] = q[self.grid.qmap.index('h_2')]
            n['h'] = n_H
            
        if 2 in self.grid.Z:
            n_He = self.grid.element_abundances[1] * n_H
            n['he'] = n_He
            x['he_1'] = q[self.grid.qmap.index('he_1')]
            x['he_2'] = q[self.grid.qmap.index('he_2')]
            x['he_3'] = q[self.grid.qmap.index('he_3')]
                
        n_e = q[self.grid.qmap.index('e')] * n_H
        
        return x, n, n_e
        
    def RateEquations(self, t, q, args):
        """
        Compute right-hand side of rate equation ODEs.
    
        Equations 1, 2, 3 and 9 in Mirocha et al. (2012), except
        we're solving for ion fractions instead of number densities.
    
        Parameters
        ----------
        t : float
            Current time.
        q : np.ndarray
            Array of dependent variables, one per rate equation.
        args : list
            Extra information needed to compute rates.
            [cell #, ionization rate coefficient (IRC), secondary IRC,
             photo-heating rate coefficient, number density, time]
        """       
    
        self.q = q
    
        cell, G, g, H, n, time = args
            
        to_temp = 1. / (1.5 * n * k_B)
    
        if self.grid.expansion:
            z = self.grid.cosm.TimeToRedshiftConverter(0., time, self.grid.zi)
            n_H = self.grid.cosm.nH(z)
        else:
            n_H = self.grid.n_H[cell]

        if 2 in self.grid.Z:
            y = self.grid.element_abundances[1]
            n_He = self.grid.element_abundances[1] * n_H
        else:
            y = 0.0
            n_He = 0.0

        # Read q vector quantities into dictionaries
        x, n, n_e = self._parse_q(q, n_H)
        
        xe = n_e / n_H

        # Initialize dictionaries for results
        k_H = {sp:H[i] for i, sp in enumerate(self.grid.absorbers)}
        Gamma = {sp:G[i] for i, sp in enumerate(self.grid.absorbers)}
        gamma = {sp:g[i] for i, sp in enumerate(self.grid.absorbers)}
        Beta = {sp:self.Beta[...,i] for i, sp in enumerate(self.grid.absorbers)}
        alpha = {sp:self.alpha[...,i] for i, sp in enumerate(self.grid.absorbers)}
        zeta = {sp:self.zeta[...,i] for i, sp in enumerate(self.grid.absorbers)}
        eta = {sp:self.eta[...,i] for i, sp in enumerate(self.grid.ions)}
        psi = {sp:self.psi[...,i] for i, sp in enumerate(self.grid.absorbers)}
        
        if 2 in self.grid.Z:
            xi = self.xi
            omega = self.omega
        
        # Store results here
        dqdt = {field:0.0 for field in self.grid.evolving_fields}
        
        ##
        # Secondary ionization (of hydrogen)
        ##
        gamma_HI = 0.0
        if self.secondary_ionization > 0:
            
            for j, donor in enumerate(self.grid.absorbers):
                elem = self.grid.parents_by_ion[donor]
                
                term = gamma['h_1'][j] * x[donor] / x['h_1']
                dqdt['h_1'] -= term
                dqdt['h_2'] += term

                gamma_HI += term

        ##
        # Hydrogen rate equations
        ##
        dqdt['h_1'] = -(Gamma['h_1'] + gamma_HI + Beta['h_1'][cell] * n_e) \
                      * x['h_1'] \
                      + alpha['h_1'][cell] * n_e * x['h_2']
        dqdt['h_2'] = -dqdt['h_1']    

        ##
        # Heating & cooling
        ##

        # NOTE: cooling term multiplied by electron density at the very end!

        heat = 0.0
        cool = 0.0
        if not self.isothermal:
            
            for sp in self.grid.neutrals:
                elem = self.grid.parents_by_ion[sp]
                
                heat += k_H[sp] * x[sp] * n[elem]         # photo-heating
            
                cool += zeta[sp][cell] * x[sp] * n[elem]  # ionization
                cool += psi[sp][cell] * x[sp] * n[elem]   # excitation

            for sp in self.grid.ions:
                elem = self.grid.parents_by_ion[sp]
                
                cool += eta[sp][cell] * x[sp] * n[elem]   # recombination
                                            
        ##
        # Helium processes
        ##
        if 2 in self.grid.Z:
            
            # Secondary ionization  
            gamma_HeI = 0.0
            gamma_HeII = 0.0 
            if self.secondary_ionization > 0: 
                                
                for j, donor in enumerate(self.grid.absorbers):
                    elem = self.grid.parents_by_ion[donor]
                
                    term1 = gamma['he_1'][j] * x[donor] / x['he_1']
                    dqdt['he_1'] -= term1
                    dqdt['he_2'] += term1
                    
                    gamma_HeI += term1
                    
                    term2 = gamma['he_2'][j] * x[donor] / x['he_2']
                    dqdt['he_2'] -= term2
                    dqdt['he_3'] += term2
                    
                    gamma_HeII += term2
            
            ##
            # Helium rate equations
            ##
            dqdt['he_1'] = \
                - x['he_1'] * (Gamma['he_1'] + gamma_HeI + Beta['he_1'][cell] * n_e) \
                + x['he_2'] * (alpha['he_1'][cell] + xi[cell]) * n_e
            
            dqdt['he_2'] = \
                  x['he_1'] * (Gamma['he_1'] + gamma_HeI + Beta['he_1'][cell] * n_e) \
                - x['he_2'] * (Gamma['he_2'] + gamma_HeII \
                + (Beta['he_2'][cell] + alpha['he_1'][cell] + xi[cell]) * n_e) \
                + x['he_3'] * alpha['he_2'][cell] * n_e
        
            dqdt['he_3'] = \
                  x['he_2'] * (Gamma['he_2'] + gamma_HeII + Beta['he_2'][cell] * n_e) \
                - x['he_3'] * alpha['he_2'][cell] * n_e
            
            # Dielectronic recombination cooling    
            if not self.grid.isothermal:
                cool += omega[cell] * x['he_2'] * n_He
                    
        ##            
        # Electrons
        ##
        
        # Gains from ionizations of HI
        dqdt['e'] = 1. * dqdt['h_2']
        
        # Electrons from helium ionizations
        if 2 in self.grid.Z:
            # Gains from ionization of HeI
            dqdt['e'] += y * x['he_1'] \
                * (Gamma['he_1'] + gamma_HeI + Beta['he_1'][cell] * n_e)
            
            # Gains from ionization of HeII
            dqdt['e'] += y * x['he_2'] \
                * (Gamma['he_2'] + gamma_HeII + Beta['he_2'][cell] * n_e)
                
            # Losses from HeII recombinations
            dqdt['e'] -= y * x['he_2'] \
                * (alpha['he_1'][cell] + xi[cell]) * n_e
                
            # Losses from HeIII recombinations
            dqdt['e'] -= y * x['he_3'] * alpha['he_2'][cell] * n_e
            
        # Finish heating and cooling
        if not self.grid.isothermal:
            hubcool = 0.0
            compton = 0.0
            
            # Hubble Cooling
            if self.grid.expansion:
                hubcool = 2. * self.grid.cosm.HubbleParameter(z) * q[-1]

                # Compton cooling
                if self.grid.compton_scattering:
                    Tcmb = self.grid.cosm.TCMB(z)
                    ucmb = self.grid.cosm.UCMB(z)
                    tcomp = 3. * m_e * c / (8. * sigma_T * ucmb)
                    compton = xe * (Tcmb - q[-1]) / tcomp \
                        / (1. + self.grid.cosm.y + xe)
            
            dqdt['Tk'] = (heat - n_e * cool) * to_temp + compton - hubcool
            
        else:
            dqdt['Tk'] = 0.0 
            
        self.dqdt = np.array([dqdt[sp] for sp in self.grid.qmap])

        return self.dqdt
                          
    def Jacobian(self, t, q, args):
        self.q = q
        self.dqdt = np.zeros_like(self.zeros_q)
    
        cell, G, g, H, n, time = args
        
        to_temp = 1. / (1.5 * n * k_B)
    
        if self.grid.expansion:
            z = self.grid.cosm.TimeToRedshiftConverter(0., time, self.grid.zi)
            n_H = self.grid.cosm.nH(z)
        else:
            n_H = self.grid.n_H[cell]
                    
        # Read q vector quantities into dictionaries
        x, n, n_e = self._parse_q(q, n_H)
        
        if 2 in self.grid.Z:
            n_He = self.grid.element_abundances[1] * n_H
        else:
            n_He = 0.0
    
        xe = n_e / n_H
    
        # Initialize dictionaries for results        
        k_H = {sp:H[i] for i, sp in enumerate(self.grid.absorbers)}
        Gamma = {sp:G[i] for i, sp in enumerate(self.grid.absorbers)}
        gamma = {sp:g[i] for i, sp in enumerate(self.grid.absorbers)}
        Beta = {sp:self.Beta[...,i] for i, sp in enumerate(self.grid.absorbers)}
        alpha = {sp:self.alpha[...,i] for i, sp in enumerate(self.grid.absorbers)}
        zeta = {sp:self.zeta[...,i] for i, sp in enumerate(self.grid.absorbers)}
        eta = {sp:self.eta[...,i] for i, sp in enumerate(self.grid.ions)}
        psi = {sp:self.psi[...,i] for i, sp in enumerate(self.grid.absorbers)}
        xi = self.xi
        omega = self.omega
    
        # For Jacobian
        if not self.isothermal:
            dBeta = {sp:self.dBeta[...,i] for i, sp in enumerate(self.grid.absorbers)}
            dalpha = {sp:self.alpha[...,i] for i, sp in enumerate(self.grid.absorbers)}
            dzeta = {sp:self.dzeta[...,i] for i, sp in enumerate(self.grid.absorbers)}
            deta = {sp:self.deta[...,i] for i, sp in enumerate(self.grid.ions)}
            dpsi = {sp:self.dpsi[...,i] for i, sp in enumerate(self.grid.absorbers)}
    
            if 2 in self.grid.Z:
                dxi = self.dxi
                domega = self.domega
    
        N = len(self.grid.evolving_fields)
        J = np.zeros([N] * 2)     
        
        # Where do the electrons live?
        if N == 6:
            e = -1
        elif N == 7:
            e = -2
        else:
            e = 2
            
        ##
        # Secondary ionization (of hydrogen)
        ##
        gamma_HI = 0.0
        if self.secondary_ionization > 0:
        
            for j, donor in enumerate(self.grid.absorbers):
                elem = self.grid.parents_by_ion[donor]
                term = gamma['h_1'][j] * x[donor] / x['h_1']
                gamma_HI += term    
        
        ##
        # FIRST: HI and HII terms. Always in slots 0 and 1.
        ##
            
        # HI by HI
        J[0,0] = -(Gamma['h_1'] + gamma_HI + Beta['h_1'][cell] * n_e)
        
        # HI by HII
        J[0,1] = alpha['h_1'][cell] * n_e
         
        # HII by HI
        J[1,0] = -J[0,0]
        
        # HII by HII
        J[1,1] = -J[0,1]
                    
        ##    
        # Hydrogen-Electron terms
        ##
        
        J[0,e] = -Beta['h_1'][cell] * x['h_1'] + alpha['h_1'][cell] * x['h_2']
        J[1,e] = -J[0,e]
        J[e,e] = n_H * J[1,e]
           
        J[e,0] = n_H * (Gamma['h_1'] + gamma_HI + Beta['h_1'][cell] * n_e)
        J[e,1] = -n_H * alpha['h_1'][cell] * n_e
        
        #if self.grid.expansion:
        #    J[e,e] -= 3. * self.grid.cosm.HubbleParameter(z)
            
        ###
        ## HELIUM INCLUDED CASES: N=6 (isothermal), N=7 (thermal evolution)   
        ###
        if N in [6, 7]:
            
            # Secondary ionization  
            gamma_HeI = 0.0
            gamma_HeII = 0.0 
            if self.secondary_ionization > 0: 
                                
                for j, donor in enumerate(self.grid.absorbers):
                    elem = self.grid.parents_by_ion[donor]
                
                    term1 = gamma['he_1'][j] * x[donor] / x['he_1']
                    gamma_HeI += term1
                    
                    term2 = gamma['he_2'][j] * x[donor] / x['he_2']
                    gamma_HeII += term2
            
            # HeI by HeI
            J[2,2] = -(Gamma['he_1'] + gamma_HeI + Beta['he_1'][cell] * n_e)
            
            # HeI by HeII
            J[2,3] = (alpha['he_1'][cell] + xi[cell]) * n_e
            
            # HeI by HeIII
            J[2,4] = 0.0

            # HeII by HeI
            J[3,2] = -J[2,2]

            # HeII by HeII
            J[3,3] = -(Gamma['he_2'] + gamma_HeII) \
                   - (Beta['he_2'][cell] + alpha['he_1'][cell] + xi[cell]) * n_e

            # HeII by HeIII
            J[3,4] = alpha['he_2'][cell] * n_e
            
            # HeIII by HeI
            J[4,2] = 0.0

            # HeIII by HeII
            J[4,3] = Gamma['he_2'] + Beta['he_2'][cell] * n_e

            # HeIII by HeIII
            J[4,4] = -alpha['he_2'][cell] * n_e

            ##
            # Helium-Electron terms
            ##

            J[2,e] = -Beta['he_1'][cell] * x['he_1'] \
                   + (alpha['he_1'][cell] + xi[cell]) * x['he_2']
            J[3,e] = Beta['he_1'][cell] * x['he_1'] \
                   - (Beta['he_2'][cell] + alpha['he_1'][cell] + xi[cell]) * x['he_2'] \
                   + alpha['he_2'][cell] * x['he_3']
            J[4,e] = Beta['he_2'][cell] * x['he_2'] - alpha['he_2'][cell] * x['he_3']
            
            J[e,2] = n_He \
                * (Gamma['he_1'] + gamma_HeI + Beta['he_1'][cell] * n_e)
            
            J[e,3] = n_He \
                * ((Gamma['he_2'] + gamma_HeII + Beta['he_2'][cell] * n_e) \
                - (alpha['he_1'][cell] + xi[cell]) * n_e)
            
            J[e,4] = -n_He * alpha['he_2'][cell] * n_e
            
            # Electron-electron terms (increment from H-only case)
            J[e,e] += n_He * x['he_1'] * Beta['he_1'][cell]
            
            J[e,e] += n_He \
                * (x['he_2'] \
                * ((Beta['he_2'][cell] - (alpha['he_1'][cell] + xi[cell]))) \
                - x['he_3'] * alpha['he_2'][cell])
                
        ##
        # Heating/Cooling from here onwards
        ##
        if self.isothermal:
            return J              

        ##
        # Hydrogen derivatives wrt Tk
        ##
        
        # HI by Tk
        J[0,-1] = -n_e * x['h_1'] * dBeta['h_1'][cell] \
                +  n_e * x['h_2'] * dalpha['h_1'][cell]
        # HII by Tk
        J[1,-1] = -J[0,-1]
        
        ##
        # Helium derivatives wrt Tk
        ##
        if 2 in self.grid.Z:  
            # HeI by Tk
            J[2,-1] = -n_e * (x['he_1'] * dBeta['he_1'][cell] \
                    - x['he_2'] * (dalpha['he_1'][cell] + dxi[cell]))

            # HeII by Tk
            J[3,-1] = -n_e * (x['he_2'] * (dBeta['he_2'][cell] \
                    + dalpha['he_1'][cell] + dxi[cell]) \
                    - x['he_3'] * dalpha['he_2'][cell])

            # HeIII by Tk
            J[4,-1] = n_e * (x['he_2'] * dBeta['he_2'][cell] \
                - x['he_3'] * dalpha['he_2'][cell])

        ##
        # Electron by Tk terms
        ##
        J[e,-1] = n_H * n_e \
            * (x['h_1'] * dBeta['h_1'][cell] - x['h_2'] * dalpha['h_1'][cell])
        
        ##
        # A few last Tk by Tk and Tk by electron terms (dielectronic recombination)
        ##
        if 2 in self.grid.Z:            
            J[e,-1] += n_He * n_e \
                * (dBeta['he_1'][cell] * x['he_1'] + dBeta['he_2'][cell] * x['he_2'] \
                - (dalpha['he_1'][cell] + dxi[cell]) * x['he_2'] \
                - dalpha['he_2'][cell] * x['he_3'])
                
        ##
        # Tk derivatives wrt neutrals
        ##
        for sp in self.grid.neutrals:
            i = self.grid.qmap.index(sp)
            elem = self.grid.parents_by_ion[sp]

            # Photo-heating
            J[-1,i] += n[elem] * k_H[sp]
            
            # Collisional ionization cooling
            J[-1,i] -= n[elem] * zeta[sp][cell] * n_e
            
            # Collisional excitation cooling
            J[-1,i] -= n[elem] * psi[sp][cell] * n_e
            
        ##
        # Tk derivatives wrt ions (only cooling terms)
        ##
        for sp in self.grid.ions:
            i = self.grid.qmap.index(sp)
            elem = self.grid.parents_by_ion[sp]

            # Recombination cooling
            J[-1,i] -= n[elem] * eta[sp][cell] * n_e
            
        # Dielectronic recombination term
        if 2 in self.grid.Z:
            J[-1,3] -= n_He * omega[cell] * n_e
        
        ##
        # Tk by Tk terms and Tk by electron terms
        ##
        for sp in self.grid.absorbers:       
            elem = self.grid.parents_by_ion[sp]
            J[-1,-1] -= n_e * n[elem] * x[sp] * dzeta[sp][cell]
            J[-1,-1] -= n_e * n[elem] * x[sp] * dpsi[sp][cell]
            
            J[-1,e] -= n[elem] * x[sp] * zeta[sp][cell]
            J[-1,e] -= n[elem] * x[sp] * psi[sp][cell]
        
        for sp in self.grid.ions:       
            elem = self.grid.parents_by_ion[sp] 
            J[-1,-1] -= n_e * n[elem] * x[sp] * deta[sp][cell]
                
            J[-1,e] -= n[elem] * x[sp] * eta[sp][cell]
         
        # Dielectronic recombination term
        if 2 in self.grid.Z: 
            J[-1,e] -= omega[cell] * n_He * x['he_2']      
            J[-1,-1] -= n_e * x['he_2'] * n_He * domega[cell]
            
        # So far, everything in units of energy, must convert to temperature    
        J[-1,:] *= to_temp    

        # Cosmological effects
        if self.grid.expansion:
            
            # These terms have the correct units already
            J[-1,-1] -= 2. * self.grid.cosm.HubbleParameter(z)

            if self.grid.compton_scattering:
                Tcmb = self.grid.cosm.TCMB(z)
                ucmb = self.grid.cosm.UCMB(z)
                tcomp = 3. * m_e * c / (8. * sigma_T * ucmb)

                J[-1,-1] -= q[-1] * xe \
                    / tcomp / (1. + self.grid.cosm.y + xe)
                J[-1,e] -= (Tcmb - q[-1]) * (1. + self.grid.cosm.y) \
                    / (1. + self.grid.cosm.y + xe)**2 / tcomp

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
        
        if not self.isothermal:
            self.dBeta = np.zeros_like(self.grid.zeros_grid_x_absorbers)
            self.dalpha = np.zeros_like(self.grid.zeros_grid_x_absorbers)
            self.dzeta = np.zeros_like(self.grid.zeros_grid_x_absorbers)
            self.deta = np.zeros_like(self.grid.zeros_grid_x_absorbers)
            self.dpsi = np.zeros_like(self.grid.zeros_grid_x_absorbers)
        
        for i, absorber in enumerate(self.grid.absorbers):
            self.Beta[...,i] = self.coeff.CollisionalIonizationRate(i, T)
            self.alpha[...,i] = self.coeff.RadiativeRecombinationRate(i, T)
            
            if self.isothermal:
                continue
                
            self.zeta[...,i] = self.coeff.CollisionalIonizationCoolingRate(i, T)
            self.eta[...,i] = self.coeff.RecombinationCoolingRate(i, T)
            self.psi[...,i] = self.coeff.CollisionalExcitationCoolingRate(i, T)
            
            self.dBeta[...,i] = self.coeff.dCollisionalIonizationRate(i, T)
            self.dalpha[...,i] = self.coeff.dRadiativeRecombinationRate(i, T)
            self.dzeta[...,i] = self.coeff.dCollisionalIonizationCoolingRate(i, T)
            self.deta[...,i] = self.coeff.dRecombinationCoolingRate(i, T)
            self.dpsi[...,i] = self.coeff.dCollisionalExcitationCoolingRate(i, T)

        # Di-electric recombination
        if 2 in self.grid.Z:
            self.xi = self.coeff.DielectricRecombinationRate(T)
            self.dxi = self.coeff.dDielectricRecombinationRate(T)
            
            if not self.isothermal:
                self.omega = self.coeff.DielectricRecombinationCoolingRate(T)
                self.domega = self.coeff.dDielectricRecombinationCoolingRate(T)
                        
        return {'Beta': self.Beta, 'alpha': self.alpha,
                'zeta': self.zeta, 'eta': self.eta, 'psi': self.psi,
                'xi': self.xi, 'omega': self.omega}

class DengoChemicalNetwork:
    def __init__(self, grid):
        self.grid = grid
        
        # Make list of dengo.chemical_network.ChemicalNetwork objects
        self._initialize()
    
    def _initialize(self):
        """ Create list of ChemicalNetwork objects (one for each element). """
        
        if not have_dengo:
            raise ImportError('Module dengo not found.')
        
        self.networks = {}
        for element in self.grid.ions_by_parent.keys():
            chem_net = ChemicalNetwork()
            for ion in self.grid.ions_by_parent[element]:
                dengo_name = convert_ion_name(ion)
                
                # Do something special!?
                #if self._is_primordial(element):
                #    continue
                
                # Add cooling actions - even if isothermal
                for ca in cooling_registry.values():
                    if ca.name.split('_')[0] == dengo_name:
                        chem_net.add_cooling(ca)
                        
                # Add ionization reactions        
                for s in reaction_registry.values():
                    if s.name.split('_')[0] == dengo_name:
                        chem_net.add_reaction(s)
                                    
            self.networks[element] = chem_net
        
        # For each chemical network object, set rate equation and Jacobian
        self._set_rhs()
        self._set_jac()
    
    def _is_primordial(self, element):
        if element in ['h', 'he']:
            return True
        else:
            return False
            
    def _set_rhs(self):
        """
        Create string representations of RHS of all ODEs.
        
        NOTE: This will not work with primordial network naming conventions.
        """
        
        # Set up RHS of rate equations - must convert dengo strings to ours
        i = 0
        dedot = ''
        gedot = ''
        self.dqdt_eqs = []
        for element in self.grid.elements:
            for ion in self.grid.ions_by_parent[element]:
                expr = self.networks[element].rhs_string_equation(convert_ion_name(ion))
                expr = self._translate_rate_str(expr)                
                self.dqdt_eqs.append(expr)        
                i += 1
                
            de = self._translate_rate_str(self.networks[element].rhs_string_equation('de'))   
            dedot += '+%s' % de  

            if not self.grid.isothermal:
                ge = self._translate_cool_str(self.networks[element].cooling_string_equation())               
                gedot += '+%s' % ge  
        
        # Electrons and gas energy
        self.dqdt_eqs.append(dedot)
        if not self.grid.isothermal:
            self.dqdt_eqs.append(gedot)
                        
    def _set_jac(self):
        """
        Create string representations of Jacobian.
        """        
        
        # Create empty jacobian
        self.jac = [['0']*len(self.grid.evolving_fields) \
            for sp in self.grid.evolving_fields]
            
        for i, sp1 in enumerate(self.grid.evolving_fields):
            
            for element in self.grid.ions_by_parent:
                if sp1 in self.grid.ions_by_parent[element]:
                    break
                    
            chemnet = self.networks[element]
            names = [species.name for species in chemnet.required_species]
            
            for j, sp2 in enumerate(self.grid.evolving_fields):                    
                if convert_ion_name(sp2) not in names:
                    continue    
                
                expr = chemnet.jacobian_string_equation(convert_ion_name(sp1), 
                    convert_ion_name(sp2))
                expr = self._translate_rate_str(expr)
                self.jac[i][j] = expr
            
    def _translate_rate_str(self, expr):
        """
        Convert dengo style rate equation strings to our dictionary / vector 
        convention.
        """    
        
        # Ionization and recombination coefficients
        for species in self.grid.all_ions:
            sp = convert_ion_name(species)
            expr = expr.replace('%s_i' % sp, 'kwargs[\'%s_i\']' % species)
            expr = expr.replace('%s_r' % sp, 'kwargs[\'%s_r\']' % species)
        
        # Rename ions
        for element in self.grid.ions_by_parent:
            ions = copy.deepcopy(self.grid.ions_by_parent[element])
            ions.reverse()
            
            for ion in ions:
                sp = convert_ion_name(ion)
                        
                expr = expr.replace('%s' % sp,
                    'q[%i]' % list(self.grid.qmap).index(ion))
        
        # Replace de and ge         
        if self.grid.isothermal:
            expr = expr.replace('de', 'q[-1]')
        else:
            expr = expr.replace('de', 'q[-2]')
            expr = expr.replace('ge', 'q[-1]')
                                    
        return expr   
        
    def _translate_cool_str(self, expr):
        for species in self.grid.all_ions:
            sp = convert_ion_name(species)
            expr = expr.replace('%s_c_%s_c' % (sp, sp), 
                'kwargs[\'%s_c\']' % species)
        
        # Rename ions
        for element in self.grid.ions_by_parent:
            ions = copy.deepcopy(self.grid.ions_by_parent[element])
            ions.reverse()
            
            for ion in ions:
                sp = convert_ion_name(ion)
                        
                expr = expr.replace('%s' % sp,
                    'q[%i]' % list(self.grid.qmap).index(ion))
        
        # Replace de and ge         
        if self.grid.isothermal:
            expr = expr.replace('de', 'q[-1]')
        else:
            expr = expr.replace('de', 'q[-2]')
            expr = expr.replace('ge', 'q[-1]')
                                    
        return expr   
        
    def RateEquations(self, t, q, args):
        """
        Compute RHS of rate equations.  
        """
        
        kwargs = dict(args)
        self.q = q
        self.dqdt = np.zeros(len(self.dqdt_eqs))
                        
        # Compute RHS of ODEs
        for i, ode in enumerate(self.dqdt_eqs):
            self.dqdt[i] = eval(ode)
                
        return self.dqdt
        
    def Jacobian(self, t, q, args):
        """
        Compute the Jacobian of the rate equations.
        """    
                
        kwargs = dict(args)
        jac = np.zeros([len(self.dqdt_eqs)] * 2)

        for i, sp1 in enumerate(self.grid.evolving_fields):
            for j, sp2 in enumerate(self.grid.evolving_fields):
                jac[i,j] = eval(self.jac[i][j])
                
        return jac                      