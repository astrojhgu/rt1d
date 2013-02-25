"""

Optimize.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Feb 25 09:00:27 2013

Description: 

"""

import rt1d, copy
import numpy as np

try:
    import ndmin
except ImportError:
    pass
    
erg_per_ev = rt1d.physics.Constants.erg_per_ev 

class Optimization:
    def __init__(self, logN=None, Z=[1], nfreq=1, rs=None, fn=None, 
        thin=False, isothermal=False, secondary_ionization=0, mcmc=False):
        self.logN = logN
        self.Z = Z
        self.nfreq = nfreq
        self.rs = rs
        self.fn = fn
        self.thin = thin
        self.isothermal = isothermal
        self.secondary_ionization = secondary_ionization
        self.mcmc = mcmc
        
        # Use Grid class to carry info about absorbing species
        self.grid = rt1d.Grid()
        self.grid.set_chem(Z=self.Z, isothermal=self.isothermal, 
            secondary_ionization=self.secondary_ionization)
        
        # Initialize radiation source
        if self.rs is None:
            self.rs = rt1d.sources.RadiationSourceIdealized(self.grid, 
                self.logN, **{'spectrum_file': self.fn})
        elif type(self.rs) is dict:
            self.rs = rt1d.sources.RadiationSourceIdealized(self.grid, 
                self.logN, **self.rs)        
        
        # What integrals are we comparing to?
        self.integrals = copy.deepcopy(self.rs.tab.IntegralList)
        self.integrals.pop(self.integrals.index('Tau'))
        
        # Figure out size of optical depth array [np.prod(logN.shape) x nfreq]
        self.tau_dims = list(self.rs.tab.dimsN.copy())
        self.tau_dims.append(len(self.Z))
        self.tau_dims.append(self.nfreq)
        
        self.tab_dims = list(self.rs.tab.dimsN.copy())
        #self.tab_dims.insert(0, len(self.Z))
        #self.tab_dims.insert(0, len(self.integrals))
    
    def run_(self, steps, limits=None, step=None):
        self.__call__(steps, limits=limits, step=step)
    
    def __call__(self, steps, guess=None, limits=None, step=None):
        """
        Construct optimal discrete spectrum for a given radiation source.
        
        Need: Column density range for all species.
            logN = list of arrays, each element should be logN for corresponding
                entry in Z.
        """

        # Initialize annealer - generous control parameters
        if limits is None:
            limits = [(13.6, 1e2)] * self.nfreq
            limits.extend([(0.0, 1.0)] * self.nfreq)
        if step is None:    
            step = [5.0] * self.nfreq
            step.extend([0.1] * self.nfreq)
            
        self.sampler = ndmin.Annealer(self.cost, limits = limits, step = step, 
            afreq = 1000)
            
        # Minimize 'func' and save results
        self.sampler.run(steps)
        
    def cost(self, pars):
        E, LE = ndmin.util.split_list(pars)
                
        # Compute optical depth for all combinations of column densities
        tau = self.tau(E)            
        
        # Compute discrete versions of phi & psi
        # NOTE: if ionization thresholds all below smallest emission energy,
        # integrals for all species are identical.
        
        discrete_tables = self.discrete_tabs(E, LE, tau)
        
        # Compute cost
        cost = 0.0
        for i, integral in enumerate(self.integrals):
            for j, absorber in enumerate(self.grid.absorbers):
                name = self.tab_name(integral, absorber)
                ref = np.log10(10**self.rs.tabs[name] * erg_per_ev)
                tmp = discrete_tables[name]
                
                cost += np.max(np.abs(tmp - ref)) + np.mean(np.abs(tmp - ref))
                
        return cost
    
    def tab_name(self, integral, absorber):
        return 'log%s_%s' % (integral, absorber)

    def discrete_tabs(self, E, LE, tau=None):
        """
        Compute values of integral quantities assuming discrete emission.
        """
        
        if tau is None:
            tau = self.tau(E)
        
        discrete_tables = {}
        for i, integral in enumerate(self.integrals):
            for j, absorber in enumerate(self.grid.absorbers):
                Eth = self.grid.ioniz_thresholds[absorber]
                
                tmp = np.zeros(self.tab_dims)
                if integral == 'Phi':
                    tmp = self.DiscretePhi(E, LE, tau[:,j,:], Eth)
                if integral == 'Psi':
                    tmp = self.DiscretePsi(E, LE, tau[:,j,:], Eth)
                        
                discrete_tables[self.tab_name(integral, absorber)] = tmp.copy()        
                            
        return discrete_tables   
        
    def DiscretePhi(self, E, LE, tau_E, Eth):
        to_sum = LE * np.exp(-tau_E) / E
        to_sum[E < Eth] = 0.0
        summed = np.sum(to_sum, axis = -1)
        return np.log10(summed)

    def DiscretePsi(self, E, LE, tau_E, Eth):
        to_sum = LE * np.exp(-tau_E)
        to_sum[E < Eth] = 0.0
        summed = np.sum(to_sum, axis = -1)
        return np.log10(summed)                     
    
    def tau(self, E):
        """
        Compute total optical depth (over all species) as a function of 
        discrete emission energy E (eV).
        """
        tau_E = np.zeros(self.tau_dims)
        for i in xrange(self.rs.tab.elements_per_table):
            for j, absorber in enumerate(self.grid.absorbers):
                loc = list(self.rs.tab.indices_N[i])
                loc.append(j)
                                    
                tau_E[tuple(loc)] = \
                    np.array(self.rs.tab.PartialOpticalDepth(E, 
                    self.rs.tab.Nall[i], absorber))
                    
        return tau_E
                