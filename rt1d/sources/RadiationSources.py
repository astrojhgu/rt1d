"""

RadiationSources.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sun Jul 22 16:22:12 2012

Description: Container for several RadiationSource____ instances.  Will loop
over said instances in ionization and heating rate calculations.

"""

import re
from ..util import parse_kwargs
from collections import Iterable
from .RadiationSource import RadiationSource

class RadiationSources:
    def __init__(self, grid=None, logN=None, init_tabs=True, **kwargs):
        self.pf = kwargs.copy()
        self.grid = grid
        
        if type(self.pf['source_type']) is not list:
            self.pf['source_type'] = [self.pf['source_type']]
            
        try:    
            self.Ns = len(self.pf['source_type'])
        except TypeError:
            self.Ns = 1
        
        if type(init_tabs) is bool:
            init_tabs = [init_tabs] * self.Ns
        
        self.all_sources = self.src = self.initialize_sources(init_tabs)
        
    def initialize_sources(self, init_tabs=True):
        """ Construct list of RadiationSource class instances. """    
        
        sources = []
        for i in xrange(self.Ns):
                        
            sf = self.pf.copy()
            
            # Construct spectrum_pars
            if self.pf['spectrum_pars'] is not None:
                try:
                    spars = self.pf['spectrum_pars'][i]
                except IndexError:
                    spars = self.pf['spectrum_pars']
                
                if spars is not None:
                    for par in spars:
                        sf.update({'spectrum_%s' % par: spars[par]})            
                    del sf['spectrum_pars']
                        
            # Add source pars
            for key in sf:
                if not re.search('source', key):
                    continue
                if type(sf[key]) is not list:
                    sf.update({key:sf[key]})
                    continue
                
                sf.update({key:sf[key][i]})                       
                                    
            # Create RadiationSource class instance
            rs = RadiationSource(grid=self.grid, init_tabs=init_tabs[i], **sf)
            
            sources.append(rs)
                
        return sources

            
    
        