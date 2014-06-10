"""

ChemicalNetwork.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Sep 20 13:15:30 2012

Description: ChemicalNetwork object just needs to have methods called
'RateEquations' and 'Jacobian'

"""

import copy
import numpy as np
from ..util import convert_ion_name
from ..physics.Constants import k_B, sigma_T, m_e, c, s_per_myr

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
    def __init__(self, grid, rate_src='fk96'):
        self.grid = grid
        
        from ..physics.RateCoefficients import RateCoefficients
        self.coeff = RateCoefficients(grid, rate_src=rate_src)

        self.isothermal = self.grid.isothermal

        # For convenience 
        self.zeros_q = np.zeros(len(self.grid.evolving_fields))
        self.zeros_jac = np.zeros([len(self.grid.evolving_fields)] * 2)
        
    def RateEquationsOLD(self, t, q, args):
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
        self.dqdt = np.zeros_like(self.zeros_q)
        
        cell, Gamma, gamma, k_H, n, time = args
        
        to_temp = 1. / (1.5 * n * k_B)
                
        if self.grid.expansion:
            z = self.grid.cosm.TimeToRedshiftConverter(0., time, self.grid.zi)
            n_H = self.grid.cosm.nH0 * (1. + z)**3
        else:
            n_H = self.grid.n_H[cell]
                    
        # h1, he1, etc. correspond to indices in absorbers list.
        # qh1, qhe1, etc. correspond to indices in q array
        
        if 1 in self.grid.Z:
            qh1, qh2, qe = (0, 1, 2)
            h1 = 0
            xHI = q[qh1]
            xHII = q[qh2] 
            nHI = n_H * xHI
            nHII = n_H * xHII
                
            if 2 in self.grid.Z:
                n_He = self.grid.element_abundances[1] * n_H
                
                if not self.grid.approx_helium:
                    
                    qhe1, qhe2 = (2, 3)
                    he1, he2 = (1, 2)
                    xHeI = q[qhe1]
                    xHeII = q[qhe2]

                    if 'he_3' in self.grid.ions:
                        qhe3, qe = (4, 5)
                        xHeIII = q[qhe2 + 1]
                    else:
                        qe = 4
                        xHeIII = 0.0

        elif 2 in self.grid.Z:
            qhe1, qhe2 = (2, 3)
            he1, he2 = (1, 2)
            xHeI = q[qhe1]
            xHeII = q[qhe2]
            
            if 'he_3' in self.grid.ions:
                qhe3, qe = (4, 5)
                xHeIII = q[qhe2 + 1]
            else:
                qe = 4
                xHeIII = 0.0
            
            n_He = self.grid.element_abundances[0] * n_H
        
        n_e = q[qe]
                                                                    
        # Hydrogen rate equations
        if 1 in self.grid.Z:
            self.dqdt[qh1] = \
                -1. * (Gamma[h1] + self.Beta[cell][h1] * n_e) * xHI \
                + self.alpha[cell][h1] * n_e * xHII \
                - gamma[h1][h1] * xHI
            self.dqdt[qh2] = -self.dqdt[qh1]   
            self.dqdt[qe] = self.dqdt[qh2] * n_H
            
            if self.grid.approx_helium:
                self.dqdt[qe] *= (1. + n_H / n_He)
                                
        # Helium rate equations
        if 2 in self.grid.Z and not self.grid.approx_helium:
            self.dqdt[qhe1] = \
                -1. * (Gamma[he1] + self.Beta[cell][he1] * n_e) * xHeI \
                + (self.alpha[cell][he1] + self.xi[cell]) * n_e * xHeII
                
            self.dqdt[qhe2] = -self.dqdt[qhe1]
                
            if 'he_3' in self.grid.ions:
                self.dqdt[qhe2] -= (Gamma[he2] + self.Beta[cell][he2] * n_e) * xHeII \
                    + self.alpha[cell][he2] * n_e * xHeIII
                self.dqdt[qhe3] = -(self.dqdt[qhe2] + self.dqdt[qhe1])
            
            self.dqdt[qe] += (self.dqdt[qhe2] + self.dqdt[qhe2+1]) * n_He

        # Temperature evolution
        if not self.isothermal:
            phoheat, ioncool, reccool, exccool = np.zeros(4)
            
            if 1 in self.grid.Z:
                phoheat += k_H[h1] * xHI * n_H
                ioncool += self.zeta[cell][h1] * xHI * n_H
                reccool += self.eta[cell][h1] * xHII * n_H
                exccool += self.psi[cell][h1] * xHI * n_H
            
            if 2 in self.grid.Z and not self.grid.approx_helium:
                phoheat += k_H[he1] * xHeI * n_He + k_H[he2] * xHeII * n_He
                ioncool += self.zeta[cell][he1] * xHeI * n_He \
                         + self.zeta[cell][he2] * xHeI * n_He
                reccool += self.eta[cell][he1] * xHeII * n_He \
                         + self.eta[cell][he2] * xHeIII * n_He
                exccool += self.psi[cell][he1] * xHeI * n_He \
                         + self.psi[cell][he2] * xHeII * n_He
            
            hubcool = 0.0
            compton = 0.0
            if self.grid.expansion:
                hubcool = 2. * self.grid.cosm.HubbleParameter(z) * q[-1]
                            
                if self.grid.compton_scattering:
                    Tcmb = self.grid.cosm.TCMB(z)
                    ucmb = self.grid.cosm.UCMB(z)
                    tcomp = 3. * m_e * c / (8. * sigma_T * ucmb)
                    compton = xHII * (Tcmb - q[-1]) / tcomp \
                        / (1. + self.grid.cosm.y + xHII)
                                                
            self.dqdt[-1] = phoheat * to_temp + compton \
                - n_e * (ioncool + reccool + exccool) * to_temp \
                - hubcool
                
            if 2 in self.grid.Z:
                self.dqdt[-1] -= n_e * xHeIII * n_He * self.omega[cell]

        # Multispecies : dqdt[-1] += n_e * xHeIII * n_He * omega

        return self.dqdt

    def _parse_q(self, q, n_H):
        x, n = {}, {}
        if 1 in self.grid.Z:
            x['h_1'] = q[self.grid.qmap.index('h_1')]
            x['h_2'] = q[self.grid.qmap.index('h_2')]
            n['h'] = n_H
            
        if 2 in self.grid.Z:
            n_He = self.grid.element_abundances[1] * n_H
            x['he_1'] = q[self.grid.qmap.index('he_1')]
            x['he_2'] = q[self.grid.qmap.index('he_2')]
        
            if 'he_3' in self.grid.ions:
                x['he_3'] = q[self.grid.qmap.index('he_3')]
            else:
                x['he_3'] = 0.0
                
            n['he'] = n_He  
                
        n_e = q[self.grid.qmap.index('de')]
        
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
        self.dqdt = np.zeros_like(self.zeros_q)
    
        cell, G, g, H, n, time = args
        
        # Write routine to dict-ify this shit
    
        to_temp = 1. / (1.5 * n * k_B)
    
        if self.grid.expansion:
            z = self.grid.cosm.TimeToRedshiftConverter(0., time, self.grid.zi)
            n_H = self.grid.cosm.nH0 * (1. + z)**3
        else:
            n_H = self.grid.n_H[cell]
    
        # Read q vector quantities into dictionaries
        x, n, n_e = self._parse_q(q, n_H)
        
        # Initialize dictionaries for results        
        k_H = {sp:H[i] for i, sp in enumerate(self.grid.absorbers)}
        Gamma = {sp:G[i] for i, sp in enumerate(self.grid.absorbers)}
        gamma = {sp:g[i] for i, sp in enumerate(self.grid.absorbers)}
        Beta = {sp:self.Beta[...,i] for i, sp in enumerate(self.grid.absorbers)}
        alpha = {sp:self.alpha[...,i] for i, sp in enumerate(self.grid.absorbers)}
        zeta = {sp:self.zeta[...,i] for i, sp in enumerate(self.grid.absorbers)}
        eta = {sp:self.eta[...,i] for i, sp in enumerate(self.grid.ions)}
        psi = {sp:self.psi[...,i] for i, sp in enumerate(self.grid.neutrals)}
        
        if 2 in self.grid.Z:
            xi = {'he_1':self.xi}

        heat, cool = 0.0, 0.0
                        
        # Loop over neutrals and ions, compute rates
        dqdt = {field:0.0 for field in self.grid.evolving_fields}
        for i, sp in enumerate(self.grid.evolving_fields):
            
            # Neutral species
            if sp in self.grid.absorbers:
                
                elem = self.grid.parents_by_ion[sp]
                k = self.grid.ions_by_parent[elem].index(sp)
                ion = self.grid.ions_by_parent[elem][k+1]

                # Losses via ionization processes
                dqdt[sp] -= (Gamma[sp] + Beta[sp][cell] * n_e) * x[sp]
      
                # Secondary ionization
                for j, donor in enumerate(self.grid.absorbers):
                    dqdt[sp] -= gamma[sp][j] * n[elem] * x[donor]      
                        
                # Gains via recombinations
                dqdt[sp] += alpha[sp][cell] * n_e * x[ion]
                
                if sp == 'he_1':
                    dqdt[sp] += xi[sp][cell] * n_e * x['he_2']
                
                # Heating & cooling
                if not self.grid.isothermal:
                    heat += k_H[sp] * x[sp] * n[elem]         # photo-heating
                    cool += zeta[sp][cell] * x[sp] * n[elem]  # ionization
                    cool += psi[sp][cell] * x[sp] * n[elem]   # excitation
                                        
                # Electrons
                dqdt['de'] += dqdt[sp] * n[elem]
            
            elif sp in self.grid.ions:
                elem = self.grid.parents_by_ion[sp]
                k = self.grid.ions_by_parent[elem].index(sp)
                neu = self.grid.ions_by_parent[elem][k-1]
                
                if len(self.grid.ions_by_parent[elem]) == 2:
                    dqdt[sp] = -dqdt[neu]
                else:
                    neu2 = self.grid.ions_by_parent[elem][k-2]
                    dqdt[sp] = -(dqdt[neu] + dqdt[neu2])
                
                if not self.grid.isothermal:    
                    cool += eta[sp][cell] * x[sp] * n[elem]   # recombination    
                
            else:
                continue
        
        # Finish heating and cooling
        if not self.grid.isothermal:
            hubcool = 0.0
            compton = 0.0
            if self.grid.expansion:
                hubcool = 2. * self.grid.cosm.HubbleParameter(z) * q[-1]
            
                if self.grid.compton_scattering:
                    Tcmb = self.grid.cosm.TCMB(z)
                    ucmb = self.grid.cosm.UCMB(z)
                    tcomp = 3. * m_e * c / (8. * sigma_T * ucmb)
                    compton = xHII * (Tcmb - q[-1]) / tcomp \
                        / (1. + self.grid.cosm.y + xHII)
            
            dqdt['Tk'] = (heat - n_e * cool) * to_temp + compton - hubcool
        else:
            dqdt['Tk'] = 0.0 
         
        # Electrons, heating & cooling    
        #elif species == 'de':
        #    qtmp[species] += self.dqdt[qh2] * n_H
        #    
        #else:
        #    if self.grid.isothermal:
        #        continue
        #    
        #    phoheat, ioncool, reccool, exccool = np.zeros(4)
        #    
        #    if 1 in self.grid.Z:
        #        phoheat += k_H[h1] * xHI * n_H
        #        ioncool += self.zeta[cell][h1] * xHI * n_H
        #        reccool += self.eta[cell][h1] * xHII * n_H
        #        exccool += self.psi[cell][h1] * xHI * n_H
        #    
        #    if 2 in self.grid.Z and not self.grid.approx_helium:
        #        phoheat += k_H[he1] * xHeI * n_He + k_H[he2] * xHeII * n_He
        #        ioncool += self.zeta[cell][he1] * xHeI * n_He \
        #                 + self.zeta[cell][he2] * xHeI * n_He
        #        reccool += self.eta[cell][he1] * xHeII * n_He \
        #                 + self.eta[cell][he2] * xHeIII * n_He
        #        exccool += self.psi[cell][he1] * xHeI * n_He \
        #                 + self.psi[cell][he2] * xHeII * n_He
        #    
        #    hubcool = 0.0
        #    compton = 0.0
        #    if self.grid.expansion:
        #        hubcool = 2. * self.grid.cosm.HubbleParameter(z) * q[-1]
        #    
        #        if self.grid.compton_scattering:
        #            Tcmb = self.grid.cosm.TCMB(z)
        #            ucmb = self.grid.cosm.UCMB(z)
        #            tcomp = 3. * m_e * c / (8. * sigma_T * ucmb)
        #            compton = xHII * (Tcmb - q[-1]) / tcomp \
        #                / (1. + self.grid.cosm.y + xHII)
        #    
        #    self.dqdt[-1] = phoheat * to_temp + compton \
        #        - n_e * (ioncool + reccool + exccool) * to_temp \
        #        - hubcool    
        #
        
        
        
        # Hydrogen rate equations
        #if 1 in self.grid.Z:
        #    self.dqdt[qh1] = \
        #        -1. * (Gamma[h1] + self.Beta[cell][h1] * n_e) * xHI \
        #        + self.alpha[cell][h1] * n_e * xHII \
        #        - gamma[h1][h1] * xHI
        #    self.dqdt[qh2] = -self.dqdt[qh1]   
        #    self.dqdt[qe] = self.dqdt[qh2] * n_H
        #
        #    if self.grid.approx_helium:
        #        self.dqdt[qe] *= (1. + n_H / n_He)
        #
        ## Helium rate equations
        #if 2 in self.grid.Z and not self.grid.approx_helium:
        #    self.dqdt[qhe1] = \
        #        -1. * (Gamma[he1] + self.Beta[cell][he1] * n_e) * xHeI \
        #        + (self.alpha[cell][he1] + self.xi[cell]) * n_e * xHeII
        #
        #    self.dqdt[qhe2] = -self.dqdt[qhe1]
        #
        #    if 'he_3' in self.grid.ions:
        #        self.dqdt[qhe2] -= (Gamma[he2] + self.Beta[cell][he2] * n_e) * xHeII \
        #            + self.alpha[cell][he2] * n_e * xHeIII
        #        self.dqdt[qhe3] = -(self.dqdt[qhe2] + self.dqdt[qhe1])
        #
        #    self.dqdt[qe] += (self.dqdt[qhe2] + self.dqdt[qhe2+1]) * n_He
        #
        ## Temperature evolution
        #if not self.grid.isothermal:
        #    phoheat, ioncool, reccool, exccool = np.zeros(4)
        #
        #    if 1 in self.grid.Z:
        #        phoheat += k_H[h1] * xHI * n_H
        #        ioncool += self.zeta[cell][h1] * xHI * n_H
        #        reccool += self.eta[cell][h1] * xHII * n_H
        #        exccool += self.psi[cell][h1] * xHI * n_H
        #
        #    if 2 in self.grid.Z and not self.grid.approx_helium:
        #        phoheat += k_H[he1] * xHeI * n_He + k_H[he2] * xHeII * n_He
        #        ioncool += self.zeta[cell][he1] * xHeI * n_He \
        #                 + self.zeta[cell][he2] * xHeI * n_He
        #        reccool += self.eta[cell][he1] * xHeII * n_He \
        #                 + self.eta[cell][he2] * xHeIII * n_He
        #        exccool += self.psi[cell][he1] * xHeI * n_He \
        #                 + self.psi[cell][he2] * xHeII * n_He
        #
        #    hubcool = 0.0
        #    compton = 0.0
        #    if self.grid.expansion:
        #        hubcool = 2. * self.grid.cosm.HubbleParameter(z) * q[-1]
        #
        #        if self.grid.compton_scattering:
        #            Tcmb = self.grid.cosm.TCMB(z)
        #            ucmb = self.grid.cosm.UCMB(z)
        #            tcomp = 3. * m_e * c / (8. * sigma_T * ucmb)
        #            compton = xHII * (Tcmb - q[-1]) / tcomp \
        #                / (1. + self.grid.cosm.y + xHII)
        #
        #    self.dqdt[-1] = phoheat * to_temp + compton \
        #        - n_e * (ioncool + reccool + exccool) * to_temp \
        #        - hubcool
        #
        #    if 2 in self.grid.Z:
        #        self.dqdt[-1] -= n_e * xHeIII * n_He * self.omega[cell]
        #
        # Multispecies : dqdt[-1] += n_e * xHeIII * n_He * omega
        
        self.dqdt = np.array([dqdt[sp] for sp in self.grid.qmap])
                
        return self.dqdt    
    
    def Jacobian(self, t, q, args):
        """
        Jacobian of the rate equations.
        """
                                        
        cell, Gamma, gamma, k_H, n, time = args        
                
        if self.grid.expansion:
            z = self.grid.cosm.TimeToRedshiftConverter(0., time, self.grid.zi)
            n_H = self.grid.cosm.nH0 * (1. + z)**3
        else:    
            n_H = self.grid.n_H[cell]        
                                    
        #if 1 in self.grid.Z:
        #    qh1, qh2, qe = (0, 1, 2)
        #    h1 = 0
        #    xHI = q[qh1]
        #    xHII = q[qh2]        
        #    nHI = n_H * xHI
        #    nHII = n_H * xHII
        #        
        #    if 2 in self.grid.Z:
        #        n_He = self.grid.element_abundances[1] * n_H
        #        
        #        if not self.grid.approx_helium:
        #            qhe1, qhe2, qhe3, qe = (2, 3, 4, 5)
        #            he1, he2 = (1, 2)                
        #            xHeI = q[qhe1]
        #            xHeII = q[qhe2]
        #            xHeIII = q[qhe3]
        #            nHeI = n_He * xHeI
        #            nHeII = n_He * xHeII
        #            nHeIII = n_He * xHeIII
        #        
        #elif 2 in self.grid.Z:    
        #    qhe1, qhe2, qhe3, qe = (0, 1, 2, 3)
        #    he1, he2 = (0, 1)
        #    n_He = self.grid.element_abundances[0] * n_H
        #    xHeI = q[qhe1]
        #    xHeII = q[qhe2]
        #    xHeIII = q[qhe3]
        #    nHeI = n_He * xHeI
        #    nHeII = n_He * xHeII
        #    nHeIII = n_He * xHeIII            
        
        if 1 in self.grid.Z:
            qh1, qh2, qe = (0, 1, 2)
            h1 = 0
            xHI = q[qh1]
            xHII = q[qh2] 
            nHI = n_H * xHI
            nHII = n_H * xHII
        
            if 2 in self.grid.Z:
                n_He = self.grid.element_abundances[1] * n_H
        
                if not self.grid.approx_helium:
        
                    qhe1, qhe2 = (2, 3)
                    he1, he2 = (1, 2)
                    xHeI = q[qhe1]
                    xHeII = q[qhe2]
        
                    if 'he_3' in self.grid.ions:
                        qhe3, qe = (4, 5)
                        xHeIII = q[qhe2 + 1]
                    else:
                        qe = 4
                        xHeIII = 0.0
        
        elif 2 in self.grid.Z:
            qhe1, qhe2 = (2, 3)
            he1, he2 = (1, 2)
            xHeI = q[qhe1]
            xHeII = q[qhe2]
        
            if 'he_3' in self.grid.ions:
                qhe3, qe = (4, 5)
                xHeIII = q[qhe2 + 1]
            else:
                qe = 4
                xHeIII = 0.0
        
            n_He = self.grid.element_abundances[0] * n_H
        
        n_e = q[qe]    
        
        n_e = q[qe]
            
        J = np.zeros_like(self.zeros_jac)

        # Hydrogen terms - diagonal
        if 1 in self.grid.Z:
            J[qh1][qh1] = -(Gamma[h1] + self.Beta[cell][h1] * n_e) \
                      -   self.alpha[cell][h1] * n_e \
                      -   gamma[h1][h1]
            J[qh2][qh2] = J[qh1][qh1]
            
            # Hydrogen - off-diagonal
            J[qh1][qh2] = -J[qh1][qh1]
            J[qh1][qe] = -self.Beta[cell][h1] * xHI \
                     +  self.alpha[cell][h1] * xHII
            J[qh2][qh1] = -J[qh1][qh1]
            J[qh2][qe] = -J[qh1][qe]
            
            # Electron elements
            J[qe][qh1] = Gamma[h1] * n_H + self.Beta[cell][h1] * n_e * n_H \
                     + self.alpha[cell][h1] * n_e * n_H \
                     + np.sum(gamma[h1]) * n_H
            J[qe][qh2] = -J[qe][qh2]
            J[qe][qe] = self.Beta[cell][h1] * n_H * xHI \
                    - self.alpha[cell][h1] * n_H * xHII     
                    
            # Gas energy
            if not self.grid.isothermal:                
                J[-1][qh1] = n_H * (k_H[h1] \
                          - n_e * (self.zeta[cell][h1] 
                          + self.psi[cell][h1] - self.eta[cell][h1]))
                J[-1][qh2] = -J[-1][qh1]

        if 2 in self.grid.Z and not self.grid.approx_helium:

            # First - diagonal elements
            J[qhe1][qhe1] = \
                -1. * (Gamma[he1] + self.Beta[cell][he1] * n_e) \
                - (self.alpha[cell][he1] + self.xi[cell]) * n_e
            
            if 'he_3' in self.grid.ions:
                J[qhe2][qhe2] = \
                    -1. * (Gamma[he2] + self.Beta[cell][he2] * n_e) \
                    - self.alpha[cell][he2] * n_e + J[qhe1][qhe1]
                J[qhe3][qhe3] = \
                    -1. * (Gamma[he2] + self.Beta[cell][he2] * n_e) \
                    - self.alpha[cell][he2] * n_e
            
            # Off-diagonal elements HeI
            
            if 'he_3' in self.grid.ions:
                J[qhe1][qhe3] = (Gamma[he1] + self.Beta[cell][he1] * n_e) \
                            + (self.alpha[cell][1] + self.xi[cell]) * n_e
                J[qhe1][qhe3] = (Gamma[he1] + self.Beta[cell][he1] * n_e) \
                                - (self.alpha[cell][he1] + self.xi[cell]) * n_e 
            
            J[qhe1][qe] = -self.Beta[cell][he1] * xHeI \
                      + (self.alpha[cell][he1] + self.xi[cell]) * xHeII                
            
            # Off-diagonal elements HeII
            J[qhe2][qhe1] = (Gamma[he1] + self.Beta[cell][he1] * n_e) \
                        - (self.alpha[cell][he1] + self.xi[cell]) * n_e \
                        + (self.Beta[cell][he1] + self.alpha[cell][he1] \
                        +  self.xi[cell]) * n_e
            
            if 'he_3' in self.grid.ions:
                J[qhe2][qhe3] = -J[qhe3][qhe2]
            
            # Off-diagonal elements HeIII
            if 'he_3' in self.grid.ions:
                J[qhe2][qe] = -self.alpha[cell][he2] * xHeIII
                J[qhe2][qe] += self.Beta[cell][he2] * xHeIII
                J[qhe3][qhe1] = -(Gamma[he1] + self.Beta[cell][0] * n_e) \
                               +   self.alpha[cell][he2] * n_e
                J[qhe3][qhe2] = (Gamma[he1] + self.Beta[cell][0] * n_e) \
                               +   self.alpha[cell][he2] * n_e
                J[qhe3][qe] = self.Beta[cell][he2] * xHeII \
                                - self.alpha[cell][he2] * xHeIII            
            
            # Electrons
            J[qe][qe] += self.Beta[cell][he1] * xHeI \
                    - self.alpha[cell][he1] * xHeII \
                    - self.xi[cell] * xHeII
                    
            if 'he_3' in self.grid.ions:
                J[qe][qe] += self.Beta[cell][he2] * xHeII
                J[qe][qe] -= self.alpha[cell][he2] * xHeIII
                    
            #J[qe][qhe1] = Gamma[1] - Gamma[2] \
            #          + (self.Beta[1][cell] - self.Beta[2][cell]) * n_e \
            #          + self.xi[cell] * n_e \
            #          + self.alpha[1][cell] * n_e + self.alpha[2][cell] * n_e 
            #J[qe][qhe2] = Gamma[2] - Gamma[1] \
            #          + (self.Beta[2][cell] - self.Beta[1][cell]) * n_e \
            #          - self.xi[cell] * n_e \
            #          - self.alpha[1][cell] * n_e + self.alpha[2][cell] * n_e
            #J[qe][qhe3] = -(Gamma[1] + Gamma[2]) \
            #          - (self.Beta[1][cell] + self.Beta[2][cell]) * n_e \
            #          + self.xi[cell] * n_e \
            #          + self.alpha[1][cell] * n_e - self.alpha[2][cell] * n_e
            
            J[qe][qe] *= n_He    
            
        if self.grid.expansion:
            J[-1][-1] = -2. * self.grid.cosm.HubbleParameter(z)        
            
            if self.grid.compton_scattering:
                Tcmb = self.grid.cosm.TCMB(z)
                ucmb = self.grid.cosm.UCMB(z)
                tcomp = 3. * m_e * c / (8. * sigma_T * ucmb)
                J[-1][-1] -= xHII / tcomp \
                    / (1. + self.grid.cosm.y + xHII)

        return J
                
    def JacobianNEW(self, t, q, args):
        self.q = q
        self.dqdt = np.zeros_like(self.zeros_q)
    
        cell, G, g, H, n, time = args
        
        # Write routine to dict-ify this shit
    
        to_temp = 1. / (1.5 * n * k_B)
    
        if self.grid.expansion:
            z = self.grid.cosm.TimeToRedshiftConverter(0., time, self.grid.zi)
            n_H = self.grid.cosm.nH0 * (1. + z)**3
        else:
            n_H = self.grid.n_H[cell]
    
        # Read q vector quantities into dictionaries
        x, n, n_e = self._parse_q(q, n_H)
        
        # Initialize dictionaries for results        
        k_H = {sp:H[i] for i, sp in enumerate(self.grid.absorbers)}
        Gamma = {sp:G[i] for i, sp in enumerate(self.grid.absorbers)}
        gamma = {sp:g[i] for i, sp in enumerate(self.grid.absorbers)}
        Beta = {sp:self.Beta[...,i] for i, sp in enumerate(self.grid.absorbers)}
        alpha = {sp:self.alpha[...,i] for i, sp in enumerate(self.grid.absorbers)}
        zeta = {sp:self.zeta[...,i] for i, sp in enumerate(self.grid.absorbers)}
        eta = {sp:self.eta[...,i] for i, sp in enumerate(self.grid.ions)}
        psi = {sp:self.psi[...,i] for i, sp in enumerate(self.grid.neutrals)}

        heat, cool = 0.0, 0.0
                        
        # Loop over neutrals and ions, compute rates
        J = np.zeros([len(self.grid.evolving_fields)] * 2)
        
        # Taking derivative of...
        for i, sp1 in enumerate(self.grid.evolving_fields):

            # With respect to...
            for j, sp2 in enumerate(self.grid.evolving_fields):

                if sp2 in ['Tk', 'de']:
                    continue
                    
                # Jacobian is symmetric, should make sure we don't do double duty    

                elem1 = self.grid.parents_by_ion[sp1]
                k1 = self.grid.ions_by_parent[elem1].index(sp1)
                ion1 = self.grid.ions_by_parent[elem1][k1+1]
            
                elem2 = self.grid.parents_by_ion[sp2]
                k2 = self.grid.ions_by_parent[elem2].index(sp2)
                ion2 = self.grid.ions_by_parent[elem2][k2+1]
            
                # Neutral species
                if sp1 in self.grid.absorbers:
                                        
                    # Losses via ionization processes
                    if elem1 == elem2:
                        if i == j:
                            J[i,j] -= (Gamma[sp1] + Beta[sp1][cell] * n_e)
                        else:
                            J[i,j] += (Gamma[sp1] + Beta[sp1][cell] * n_e)
                    
                    # Secondary ionization
                    #for j, donor in enumerate(self.grid.absorbers):
                    J[i,j] -= gamma[sp2][j] * n[elem2] * x[sp2]

                    # Gains via recombinations
                    if elem1 == elem2:
                        if i == j:
                            J[i,j] += alpha[sp1][cell] * n_e #* x[ion]
                        else:
                            J[i,j] -= alpha[sp1][cell] * n_e #* x[ion]
                            
                    #if sp == 'he_1':
                    #    dqdt[sp] += self.xi[cell] * n_e * x['he_2']
                    
                    # Heating & cooling
                    #if not self.grid.isothermal:
                    #    heat += k_H[sp] * x[sp] * n[elem]
                    #    cool += zeta[sp][cell] * x[sp] * n[elem]  # ionization
                    #    cool += psi[sp][cell] * x[sp] * n[elem]   # excitation
                    #                        
                    ## Electrons
                    #dqdt['de'] += dqdt[sp] * n[elem]
                
                #elif sp in self.grid.ions:
                #
                #    if len(self.grid.ions_by_parent[elem]) == 2:
                #        J[i,j] = 0.0#J[j,i]
                #    else:
                #        neu2 = self.grid.ions_by_parent[elem][k-2]
                #        dqdt[sp] = -(dqdt[neu] + dqdt[neu2])
                #    
                #    if not self.grid.isothermal:    
                #        cool += eta[sp][cell] * x[sp] * n[elem]   # recombination    

                else:
                    continue
        
        ## Finish heating and cooling
        #if not self.grid.isothermal:
        #    hubcool = 0.0
        #    compton = 0.0
        #    if self.grid.expansion:
        #        hubcool = 2. * self.grid.cosm.HubbleParameter(z) * q[-1]
        #    
        #        if self.grid.compton_scattering:
        #            Tcmb = self.grid.cosm.TCMB(z)
        #            ucmb = self.grid.cosm.UCMB(z)
        #            tcomp = 3. * m_e * c / (8. * sigma_T * ucmb)
        #            compton = xHII * (Tcmb - q[-1]) / tcomp \
        #                / (1. + self.grid.cosm.y + xHII)
        #    
        #    dqdt['Tk'] = (heat - n_e * cool) * to_temp + compton - hubcool
        #else:
        #    dqdt['Tk'] = 0.0
        
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
        if 2 in self.grid.Z and not self.grid.approx_helium:
            self.xi = self.coeff.DielectricRecombinationRate(T)
            self.omega = self.coeff.DielectricRecombinationCoolingRate(T)
        
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