"""

InitializeGrid.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Sep 20 14:18:27 2012

Description: 

"""

import copy
import numpy as np
from collections import Iterable
from ..util import parse_kwargs, rebin
from ..physics.Hydrogen import Hydrogen
from ..physics.Cosmology import Cosmology
from ..physics.Constants import k_B, cm_per_kpc, s_per_myr, m_H
from ..physics.ComputeCrossSections import PhotoIonizationCrossSection

try:
    import chianti.core as cc
    import chianti.util as util
    have_chianti = True
except ImportError:
    from ..util import fake_chianti
    util = fake_chianti()
    have_chianti = False

try:
    from periodic.table import element as ELEMENT
except ImportError:
    from ..util import ELEMENT

tiny_number = 1e-8  # A relatively small species fraction

class Grid:
    def __init__(self, dims=64, length_units=cm_per_kpc, start_radius=0.01):
        """ Initialize grid object. """
        
        self.dims = int(dims)
        self.length_units = length_units
        self.start_radius = start_radius
        
        # Compute cell centers and edges        
        self.r_edg = self.r = \
            np.linspace(self.R0, length_units, self.dims + 1)
        self.r_int = self.r_edg[0:-1]
        self.dr = np.diff(self.r_edg)
        self.r_mid = rebin(self.r_edg)
        
        self.zi = 0
                
    @property
    def zeros_absorbers(self):
        return np.zeros(self.N_absorbers)
    
    @property
    def zeros_absorbers2(self):
        return np.zeros([self.N_absorbers] * 2)    
    
    @property
    def zeros_grid_x_absorbers(self):
        return np.zeros([self.dims, self.N_absorbers])
        
    @property
    def zeros_grid_x_absorbers2(self):
        return np.zeros([self.dims, self.N_absorbers, self.N_absorbers])     
                
    @property
    def R0(self):
        """ Start radius in lenght_units. """
        return self.start_radius * self.length_units
        
    @property
    def Vsh(self):
        """ Shell volume in length_units**3. """
        if not hasattr(self, '_Vsh_all'):
            self._Vsh_all = self.ShellVolume(self.r_edg[0:-1], self.dr)
            
        return self._Vsh_all
                
    @property            
    def neutrals(self):
        """ Return list of all neutral species. """            
        if not hasattr(self, '_neutral_species'):
            self._neutral_species = []
            for element in self.elements:
                self._neutral_species.append('%s_1' % element)

        return self._neutral_species
                    
    @property            
    def ions(self):
        """ Return list of all ionized species. """     
        if not hasattr(self, '_ionized_species'):
            neutrals = self.neutrals
            self._ionized_species = []
            for ion in self.all_ions:
                if ion in neutrals:
                    continue
                
                self._ionized_species.append(ion)
                
        return self._ionized_species
    
    @property
    def absorbers(self):    
        """ Return list of absorbers (don't include electrons). """
        if not hasattr(self, '_absorbing_species'):
            self._absorbing_species = copy.copy(self.neutrals)
            for ion in self.ions_by_parent:
                self._absorbing_species.extend(self.ions_by_parent[ion][1:-1])
            
        return self._absorbing_species
        
    @property
    def N_absorbers(self):
        """ Return number of absorbing species. """
        if not hasattr(self, 'self._num_of_absorbers'):
            absorbers = self.absorbers
            self._num_of_absorbers = int(len(absorbers))
            
        return self._num_of_absorbers
        
    @property
    def metals(self):
        """ Return list of anything that is not hydrogen or helium. """
        if not hasattr(self, '_metals'):
            self._metals = []
            self._metal_ions = []
            for element in self.ions_by_parent:
                if element in ['h', 'he']:
                    continue
                     
                self._metals.append(element)
                for ion in self.ions_by_parent[element]:
                    self._metal_ions.append(ion)
            
        return self._metals  
        
    @property
    def metal_ions(self):
        """ Return list of all metal (not H or He) ions."""      
        if not hasattr(self, '_metal_ions'):
            all_metals = self.metals
            
        return self._metal_ions
        
    @property
    def species_abundances(self):
        """
        Return dictionary containing abundances of parent
        elements of all ions.
        """
        if not hasattr(self, '_species_abundances'):
            self._species_abundances = {}
            for ion in self.ions_by_parent:
                for state in self.ions_by_parent[ion]:
                    self._species_abundances[state] = \
                        self.element_abundances[self.elements.index(ion)]
    
        return self._species_abundances
        
    @property
    def types(self):
        """
        Return list (matching all_species) with integers describing
        species type:
            0 = neutral
           +1 = ion
           -1 = other
        """
        
        if not hasattr(self, '_species_types'):
            self._species_types = []
            for species in self.all_species:
                if species in self.neutrals:
                    self._species_types.append(0)
                elif species in self.ions:
                    self._species_types.append(1)
                else:
                    self._species_types.append(-1) 
        
        return self._species_types       
        
    @property # MUST GENERALIZE THIS
    def ioniz_thresholds(self):
        """
        Return dictionary containing ionization threshold energies (in eV)
        for all absorbers.
        """    
        
        if not hasattr(self, '_ioniz_thresholds'):
            self._ioniz_thresholds = {}
            for absorber in self.absorbers:
                if absorber == 'h_1':
                    self._ioniz_thresholds[absorber] = 13.6
                elif absorber == 'he_1':
                    self._ioniz_thresholds[absorber] = 24.4
                elif absorber == 'he_2':
                    self._ioniz_thresholds[absorber] = 54.4
                    
        return self._ioniz_thresholds
        
    @property # MUST GENERALIZE THIS
    def bf_cross_sections(self):
        """
        Return dictionary containing functions that compute the bound-free 
        absorption cross-sections for all absorbers.
        """    
        
        if not hasattr(self, 'all_xsections'):
            self._bf_xsections = {}
            for absorber in self.absorbers:
                #ion = cc.continuum(absorber)
                #ion.vernerCross(energy = np.logspace(1, 5, 1000))
                if absorber == 'h_1':
                    self._bf_xsections[absorber] = lambda E: \
                        PhotoIonizationCrossSection(E, species = 0)
                elif absorber == 'he_1':
                    self._bf_xsections[absorber] = lambda E: \
                        PhotoIonizationCrossSection(E, species = 1)
                
        return self._bf_xsections
        
    @property
    def x_to_n(self):
        """
        Return dictionary containing conversion factor between species
        fraction and number density for all species.
        """
        if not hasattr(self, '_x_to_n_converter'):
            self._x_to_n_converter = {}
            for ion in self.all_ions:
                self._x_to_n_converter[ion] = self.n_ref \
                    * self.species_abundances[ion]  
        
        return self._x_to_n_converter
        
    @property
    def expansion(self):
        if not hasattr(self, '_expansion'):
            self.set_physics()
        return self._expansion
    
    @property
    def isothermal(self):
        if not hasattr(self, '_isothermal'):
            self.set_physics()
        return self._isothermal
    
    @property
    def secondary_ionization(self):
        if not hasattr(self, '_secondary_ionization'):
            self.set_physics()
        return self._secondary_ionization
    
    @property
    def compton_scattering(self):
        if not hasattr(self, '_compton_scattering'):
            self.set_physics()
        return self._compton_scattering
        
    @property
    def recombination(self):
        if not hasattr(self, '_recombination'):
            self.set_physics()
        return self._recombination
        
    @property
    def hydr(self):
        if not hasattr(self, '_hydr'):
            self._hydr = Hydrogen(self.cosm)
        return self._hydr    
            
    @property
    def cosm(self):
        if not hasattr(self, '_cosm'):
            self._cosm = Cosmology()
        return self._cosm
            
    def ColumnDensity(self, data):
        """ Compute column densities for all absorbing species. """    
        
        N = {}
        Nc = {}
        logN = {}
        for absorber in self.absorbers:
            Nc[absorber] = self.dr * data[absorber] * self.x_to_n[absorber]            
            N[absorber] = np.cumsum(Nc[absorber])
            logN[absorber] = np.log10(N[absorber])
            
        return N, logN, Nc
                
    def set_physics(self, isothermal=False, compton_scattering=False,
        secondary_ionization=0, expansion=False, recombination='B'):
        self._isothermal = isothermal
        self._compton_scattering = compton_scattering
        self._secondary_ionization = secondary_ionization
        self._expansion = expansion
        self._recombination = recombination
        
        if self._expansion:
            self.set_cosmology()
        
    def set_cosmology(self, initial_redshift=1e3, OmegaMatterNow=0.272, 
        OmegaLambdaNow=0.728, OmegaBaryonNow=0.044, HubbleParameterNow=0.702, 
        HeliumFractionByMass=0.2477, CMBTemperatureNow=2.725):
        
        self.zi = initial_redshift
        self._cosm = Cosmology(OmegaMatterNow=OmegaMatterNow, 
            OmegaLambdaNow=OmegaLambdaNow, OmegaBaryonNow=OmegaBaryonNow,
            HubbleParameterNow=HubbleParameterNow, 
            HeliumFractionByMass=HeliumFractionByMass,
            CMBTemperatureNow=CMBTemperatureNow)        
        
    def set_chemistry(self, Z=1, abundance=1.0):
        """
        Initialize chemistry - which species we'll be solving for and their 
        abundances ('cosmic', 'sun_photospheric', 'sun_coronal', etc. are
        acceptable arguments if we have chianti). Creates initial (empty)
        data dictionary to population with other set_* routines.
        """                
        
        if type(Z) is not list:
            Z = [Z]
        if type(abundance) not in [str, list]:
            abundance = [abundance]    
        
        self.abundance = abundance
        
        self.Z = np.array(Z)
        self.ions_by_parent = {} # Ions sorted by parent element in dictionary
        self.elements = []       # Just a list of element names
        self.all_ions = []       # All ion species
        self.all_species = []    # Anything with an ODE we'll later solve
          
        for element in self.Z:
            element_name = util.z2element(element)
            self.ions_by_parent[element_name] = []
            self.elements.append(element_name)
            for ion in xrange(element + 1):
                name = util.zion2name(element, ion + 1)
                self.all_ions.append(name)
                self.all_species.append(name)
                self.ions_by_parent[element_name].append(name)
      
        self.all_species.append('de')
        if not self.isothermal:
            self.all_species.append('T')
            
        # Create blank data fields
        self.data = {}
        for field in self.all_species:
            self.data[field] = np.zeros(self.dims)
            
        # Read abundances from chianti
        if type(abundance) is str and have_chianti:
            self.abundances_by_number = util.abundanceRead(abundance)['abundance']
            self.element_abundances = []
            for i, Z in enumerate(self.Z):
                self.element_abundances.append(self.abundances_by_number[Z-1])
        elif type(abundance) is str:
            raise ValueError('If chianti is not installed, must supply abundances by number.')             
        else:
            self.abundances_by_number = self.abundance
            self.element_abundances = []
            for i, Z in enumerate(self.Z):
                self.element_abundances.append(self.abundances_by_number[i])
                               
        # Initialize mapping between q-vector and physical quantities (dengo)                
        self._set_qmap()

    def set_density(self, rho0=None):
        """
        Initialize gas density and from that, the hydrogen number density 
        (which normalizes all other number densities).
        """                
        
        if isinstance(rho0, Iterable):
            self.data['rho'] = rho0
        else:
            self.data['rho'] = rho0 * np.ones(self.dims)   
                    
        if len(self.Z) == 1:
            if self.Z == np.ones(1):
                self.abundances_by_number = self.element_abundances = np.ones(1)
                if self.expansion:
                    self.n_H = self.n_ref = \
                        (1. - self.cosm.Y) * self.data['rho'] / m_H
                else:
                    self.n_H = self.n_ref = self.data['rho'] / m_H    
                return
            else:
                self.n_H = self.n_ref = self.data['rho'] / m_H \
                    / ELEMENT(self.elements[0]).mass
                return    
                            
        # Set hydrogen number density (which normalizes all other species)
        X = 0.0
        for i in xrange(len(self.abundances_by_number) - 1):
            name = util.z2element(i + 1)
            if not name.strip():
                continue
                    
            ele = ELEMENT(name)
            X += self.abundances_by_number[i] * ele.mass
                     
        # Set reference number density
        if 'h' in self.elements:                           
            self.n_H = self.n_ref = self.data['rho'] / m_H / X
        else:
            self.n_H = self.n_ref = self.data['rho'] / m_H \
                / ELEMENT(self.elements[0]).mass
        
    def set_temperature(self, T0):
        """
        Set initial temperature in grid.  If type(T0) is float, assume uniform
        temperature everywhere.  Otherwise, pass T0 as an array of values.
        """
        
        if isinstance(T0, Iterable):
            self.data['T'] = T0
        else:
            self.data['T'] = T0 * np.ones(self.dims)
            
    def set_ionization(self, Z=None, x=None, state=None, perturb=0):
        """
        Set initial ionization state.  If Z is None, assume constant ion fraction 
        of 1 / (1 + Z) for all elements.  Can be overrideen by 'state', which can be
        'equilibrium', and maybe eventually other options (e.g. perturbed out of
        equilibrium slightly, perhaps).
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
                    # i.e. everything is fine if you loop over T but that's 
                    # way slower.
                    if perturb > 0:
                        tmp = self.data[name] * np.random.normal(loc=1.0, 
                            scale=perturb, size=self.dims)
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
                self.data[species].fill(1. \
                    / (1. + util.convertName(species)['Z']))
        
        # Set electron density
        self._set_electron_density()
        
    def set_ics(self, data):
        """
        Simple way of setting all initial conditions at once with a data 
        dictionary.
        """
        
        self.data = {}
        for key in data.keys():
            if type(data[key]) is float:
                self.data[key] = data[key]
                continue
                
            self.data[key] = data[key].copy()    
    
    def make_clump(self, position = None, radius = None, overdensity = None,
        temperature = None, ionization = None, profile = None):
        """ Create a clump! """
                
        # Figure out where the clump is
        gridarr = np.linspace(0, 1, self.dims)
        isclump = (gridarr >= (position - radius)) \
                & (gridarr <= (position + radius))
                
        # First, modify density and temperature
        if profile == 0:
            self.data['rho'][isclump] *= overdensity
            self.n_H[isclump] *= overdensity
            self.n_ref[isclump] *= overdensity
            self.data['T'][isclump] = temperature
        #if profile == 1:
        #    self.data['rho'] += self.data['rho'] * overdensity \
        #        * np.exp(-(gridarr - position)**2 / 2. / radius**2)
        #    self.n_H += self.n_H * overdensity \
        #        * np.exp(-(gridarr - position)**2 / 2. / radius**2)
        #    self.data['T'] -= self.data['T'] * overdensity \
        #        * np.exp(-(gridarr - position)**2 / 2. / radius**2)
           
        # Need to think more about Gaussian clump T, x.   
                
        # Ionization state - could generalize this more
        for neutral in self.neutrals:
            self.data[neutral][isclump] = 1. - ionization
        for ion in self.ions:
            self.data[ion][isclump] = ionization    
        
        # Reset electron density, particle density, and gas energy
        self._set_electron_density()
                
        del self._x_to_n_converter
        
    def _set_electron_density(self):
        """
        Set electron density - must have run set_density beforehand.
        """
        
        self.data['de'] = np.zeros(self.dims)
        for i, Z in enumerate(self.Z):
            for j in np.arange(1, 1 + Z):   # j = number of electrons donated by ion j + 1
                x_i_jp1 = self.data[util.zion2name(Z, j + 1)]
                self.data['de'] += j * x_i_jp1 * self.n_ref \
                    * self.element_abundances[i]    
                
    def particle_density(self, data, z=0):
        """
        Compute total particle number density.
        """    
        
        n = data['de'].copy()
        for ion in self.all_ions:
            n += data[ion] * self.x_to_n[ion] * (1. + z)**3 \
                / (1. + self.zi)**3
             
        return n 
            
    def electron_density(self, data, z):
        de = np.zeros(self.dims)
        for i, Z in enumerate(self.Z):
            for j in np.arange(1, 1 + Z):   # j = number of electrons donated by ion j + 1
                x_i_jp1 = data[util.zion2name(Z, j + 1)]
                de += j * x_i_jp1 * self.n_ref * (1. + z)**3 / (1. + self.zi)**3 \
                    * self.element_abundances[i]

        return de

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

        

        