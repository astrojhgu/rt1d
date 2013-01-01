"""

InitializeGrid.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Sep 20 14:18:27 2012

Description: 

"""

import copy
import numpy as np
import chianti.core as cc
import chianti.util as util
from ..util import parse_kwargs, rebin
from periodic.table import element as ELEMENT
from ..physics.Constants import g_per_amu, k_B

tiny_number = 1e-8  # A relatively small species fraction

class Grid:
    def __init__(self, dims = 64, **kwargs):
        self.dims = int(dims)
        self.pf = parse_kwargs(**kwargs)
        
        self.r_edg = self.r = \
            np.linspace(self.R0, self.pf['LengthUnits'], self.dims + 1)
        self.dr = np.diff(self.r_edg)
        self.r_mid = rebin(self.r_edg)
        
    @property
    def zeros_absorbers(self):
        return np.zeros(len(self.absorbers))
    
    @property
    def zeros_grid_x_absorbers(self):
        return np.zeros([self.dims, len(self.absorbers)])
        
    @property
    def zeros_grid_x_absorbers2(self):
        return np.zeros([self.dims, len(self.absorbers), len(self.absorbers)])     
                
    @property
    def R0(self):
        return self.pf['StartRadius'] * self.pf['LengthUnits']    
        
    @property
    def dr(self):
        """ Return cell widths. """
        if not hasattr(self, 'dr_all'):
            self.dr_all = [np.diff(self.x) * self.pf['LengthUnits']]
            self.dr_all.append()
            
    @property
    def Vsh(self):
        if not hasattr(self, 'Vsh_all'):
            self.Vsh_all = self.ShellVolume(self.r_edg[0:-1], self.dr)
            
        return self.Vsh_all
                        
    @property            
    def neutrals(self):
        """ Return list of all neutral species. """            
        if not hasattr(self, 'neutral_species'):
            self.neutral_species = []
            for element in self.elements:
                self.neutral_species.append('%s_1' % element)

        return self.neutral_species
                    
    @property            
    def ions(self):
        """ Return list of all ionized species. """     
        if not hasattr(self, 'ionized_species'):
            neutrals = self.neutrals
            self.ionized_species = []
            for ion in self.all_ions:
                if ion in neutrals:
                    continue
                
                self.ionized_species.append(ion)
                
        return self.ionized_species
    
    @property
    def absorbers(self):    
        """ Return list of absorbers (don't include electrons). """
        if not hasattr(self, 'absorbing_species'):
            self.absorbing_species = self.neutrals
            for ion in self.ions_by_ion:
                self.absorbing_species.extend(ion[1:-1])
            
        return self.absorbing_species
        
    @property
    def species_abundances(self):
        """
        Return dictionary containing abundances of all ions' parent
        element.
        """
        if not hasattr(self, 'species_abundances'):
            self.species_abundances = {}
            for ion in self.ions_by_ion:
                for state in self.ions_by_ion[ion]:
                    self.species_abundances[state] = \
                        self.element_abundances[self.elements.index(ion)]
    
        return self.species_abundances
    
    @property
    def types(self):
        """
        Return list (matching all_species) with integers describing
        species type:
            0 = neutral
           +1 = ion
           -1 = other
        """
        
        if not hasattr(self, 'species_types'):
            self.species_types = []
            for species in self.all_species:
                if species in self.neutrals:
                    self.species_types.append(0)
                elif species in self.ions:
                    self.species_types.append(1)
                else:
                    self.species_types.append(-1) 
        
        return self.species_types   
        
    @property # MUST GENERALIZE THIS
    def ioniz_thresholds(self):
        """
        Return bound-free absorption cross-sections for all absorbers.
        """    
        
        if not hasattr(self, 'all_thresholds'):
            self.all_thresholds = []
            for absorber in self.absorbers:
                if absorber == 'h_1':
                    self.all_thresholds.append(13.6)
                elif absorber == 'he_1':
                    self.all_thresholds.append(24.4)
                elif absorber == 'he_2':
                    self.all_thresholds.append(54.4)
                    
        return self.all_thresholds
        
    @property
    def x_to_n(self):
        if not hasattr(self, 'x_to_n_converter'):
            self.x_to_n_converter = {}
            for ion in self.all_ions:
                self.x_to_n_converter[ion] = self.n_H \
                    * self.species_abundances[ion]  
        
        return self.x_to_n_converter              
        
    def set_chem(self, Z = 1, abundance = None, isothermal = False):
        """
        Initialize chemistry - which species we'll be solving for and their 
        abundances ('cosmic', 'sun_photospheric', 'sun_coronal', etc.).
        """                
        
        if type(Z) is not list:
            Z = [Z]
        
        self.abundance = abundance
        self.isothermal = isothermal
        
        self.Z = np.array(Z)
        self.ions_by_ion = {}       # Ions sorted by parent element in dictionary
        self.elements = []          # Just a list of element names
        self.all_ions = []          # All ion species
        self.all_species = []       # Anything with an ODE we'll later solve
          
        for element in self.Z:
            element_name = util.z2element(element)
            self.ions_by_ion[element_name] = []
            self.elements.append(element_name)
            for ion in xrange(element + 1):
                name = util.zion2name(element, ion + 1)
                self.all_ions.append(name)
                self.all_species.append(name)
                self.ions_by_ion[element_name].append(name)
      
        self.all_species.append('de')          
        if not isothermal:
            self.all_species.append('ge')
            
        # Create blank data fields
        self.data = {}
        for field in self.all_species:
            self.data[field] = np.zeros(self.dims)
            
        # Read abundances from chianti
        if abundance is not None:
            self.abundances_by_number = util.abundanceRead(abundance)['abundance']
        
            self.element_abundances = []
            for i, Z in enumerate(self.Z):
                self.element_abundances.append(self.abundances_by_number[Z - 1])
                       
        # Initialize mapping between q-vector and physical quantities (dengo)                
        self._set_qmap()
                        
        # Add RT fields here    

    def set_ics(self, data):
        """
        Simple way of setting all initial conditions at once with a data 
        dictionary.
        """
        
        for key in data.keys():
            self.data[key] = data[key]

    def set_T(self, T0):
        """
        Set initial temperature in grid.  If type(T0) is float, assume uniform
        temperature everywhere.  Otherwise, pass T0 as an array of values.
        """
        
        if hasattr(T0, 'size'):
            self.data['T'] = T0
        else:
            self.data['T'] = np.zeros(self.dims)
            self.data['T'].fill(T0)
                        
        # Initialize gas energy    
        if not self.isothermal:
            self.data['ge'] = 1.5 * k_B * self.particle_density(self.data) \
                / self.data['T']                
            
    def set_x(self, Z = None, x = None, state = None, perturb = 0):
        """
        Set initial ionization state.  If Z is None, assume constant ion fraction 
        of 1 / (1 + Z) for all elements.  Can be overrideen by 'state', which can be
        'equilibrium', and maybe eventually other options (e.g. perturbed out of
        equilibrium slightly perhaps).
        """       
        
        if x is not None:
            self.data[util.zion2name(Z, Z)].fill(1. - x)
            self.data[util.zion2name(Z, Z + 1)].fill(x)
            
        elif state == 'equilibrium':
            np.seterr(all = 'ignore')   # This tends to produce divide by zero errors
            for Z in self.Z:
                eq = cc.ioneq(Z, self.data['T'])
                
                for i in xrange(1 + Z):
                    mask = np.isnan(eq.Ioneq[i])
                    name = util.zion2name(Z, i + 1)
                    self.data[name][:] = eq.Ioneq[i]
                    self.data[name][mask] = np.ones_like(mask[mask == True])
                    # For some reason chianti sometimes gives nans where
                    # the neutral fraction (in oxygen at least) should be 1.
                    # It only happens when cc.ioneq is given an array of temps,
                    # i.e. everything is fine if you loop over T but that's way slower.
                                        
                    if perturb > 0:
                        tmp = self.data[name] * np.random.normal(loc = 1.0, scale = perturb, size = self.dims)        
                        tmp[tmp < tiny_number] = tiny_number
                        self.data[name] = copy.copy(tmp)
                                   
                # Renormalize                                
                if perturb > 0:
                    C = 0
                    for i in xrange(1 + Z):
                        name = util.zion2name(Z, i + 1)
                        C += self.data[name]    
                            
                    for i in xrange(1 + Z):
                        name = util.zion2name(Z, i + 1)        
                        self.data[name] /= C
                                        
            np.seterr(all = None)
            
        elif state == 'neutral':
            for Z in self.Z:                
                for i in xrange(1 + Z):
                    name = util.zion2name(Z, i + 1)
                    
                    if i == 0:
                        self.data[name] = np.ones(self.dims) - tiny_number
                    else:
                        self.data[name] = tiny_number * np.ones(self.dims)
        
        else:
            for species in self.all_ions:
                self.data[species].fill(1. / (1. + util.convertName(species)['Z']))
        
        # Set electron density
        self._set_de()
        
    def set_rho(self, rho0 = None):
        """
        Initialize gas density and from that, the hydrogen number density 
        (which normalizes all other number densities).
        """                
        
        if hasattr(rho0, 'size'):
            self.data['rho'] = rho0
        else:
            self.data['rho'] = np.zeros(self.dims)
            self.data['rho'].fill(rho0)   
                    
        if len(self.Z) == 1:
            if self.Z == np.ones(1):
                self.abundances_by_number = self.element_abundances = np.ones(1)
                self.n_H = self.data['rho'] / g_per_amu
                return
            
        # Set hydrogen number density (which normalizes all other species)
        X = 0
        for i in xrange(len(self.abundances_by_number) - 1):
            name = util.z2element(i + 1)
            if not name.strip():
                continue
                
            ele = ELEMENT(name)
            X += self.abundances_by_number[i] * ele.mass
                                                          
        self.n_H = self.data['rho'] / g_per_amu / X
        
    def particle_density(self, data):
        """
        Compute total particle number density.
        """    
        
        n = data['de'].copy()
        for ion in self.all_ions:
             n += data[ion] * self.x_to_n[ion]
             
        return n 

    def _set_de(self):
        """
        Set electron density - must have run set_rho beforehand.
        """
        
        for i, Z in enumerate(self.Z):
            for j in np.arange(1, 1 + Z):   # j = number of electrons donated by ion j + 1
                x_i_jp1 = self.data[util.zion2name(Z, j + 1)]
                self.data['de'] += j * x_i_jp1 * self.n_H * self.element_abundances[i]

    def _set_qmap(self):
        """
        The vector 'q' is an array containing the values of all ion fractions and the
        gas energy.  This routine sets up the mapping between elements in q and the
        corrresponding physical quantities.
        
        Will be in order of increasing Z, then de, then ge.
        """
        
        self.qmap = []
        for species in self.all_species:
            self.qmap.append(species)
            
    def ShellVolume(self, r, dr):
        """
        Return volume of shell at distance r, thickness dr.
        """
        
        return 4. * np.pi * ((r + dr)**3 - r**3) / 3.            

        

        