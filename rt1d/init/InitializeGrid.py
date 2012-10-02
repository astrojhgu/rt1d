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
from scipy.optimize import fsolve
from ..mods.Constants import g_per_amu
from periodic.table import element as ELEMENT

tiny_number = 1e-8  # A relatively small species fraction

class Grid:
    def __init__(self, dims = 64):
        self.dims = int(dims)
                
        # Deal with log-grid, compute dx
        #self.R0 = pf['StartRadius'] * pf['LengthUnits']
        #if pf['LogarithmicGrid']:
        #    self.r = np.logspace(np.log10(self.R0), \
        #        np.log10(pf['LengthUnits']), int(pf['GridDimensions']) + 1)
        #else:
        #    self.r = np.linspace(self.R0, pf['LengthUnits'], int(pf['GridDimensions']) + 1)
        #
        #self.dx = np.diff(self.r)   
        #self.r = self.r[0:-1]             
        #self.grid = np.arange(len(self.r))
        #self.fgrid = map(float, self.grid)      
        #                    
        ## Generic data array                
        #self.density = np.array(map(self.InitializeDensity, self.grid))
        #self.ionization = np.array(map(self.InitializeIonization, self.grid))
                    
    def set_chem(self, Z = [1], abundance = 'cosmic', isothermal = False):
        """
        Initialize chemistry - which species we'll be solving for and their 
        abundances ('cosmic', 'solar_photospheric', 'solar_coronal', etc.).
        """                
        
        self.isothermal = isothermal
        
        self.Z = np.array(Z)
        self.ions = {}              # Ions sorted by parent element in dictionary
        self.elements = []          # Just a list of element names
        self.ion_species = []       # All ion species
        self.all_species = []       # Anything with an ODE we'll later solve
        self.fields = ['T', 'de', 'rho']    # Anything we want to store for easy analysis
          
        for element in self.Z:
            element_name = util.z2element(element)
            self.ions[element_name] = []
            self.elements.append(element_name)
            for ion in xrange(element + 1):
                name = util.zion2name(element, ion + 1)
                self.fields.append(name)
                self.ion_species.append(name)
                self.all_species.append(name)
                self.ions[element_name].append(name)
      
        self.all_species.append('de')          
        if not isothermal:
            self.fields.append('ge')
            self.all_species.append('ge')
              
        # Create blank data fields        
        self.data = {}
        for field in self.fields:
            self.data[field] = np.zeros(self.dims) 
            
        # Read abundances from chianti
        self.abundances_by_number = util.abundanceRead(abundance)['abundance']
        
        self.element_abundances = []
        for i, Z in enumerate(self.Z):
            self.element_abundances.append(self.abundances_by_number[Z - 1])
                       
        # Initialize mapping between q-vector and physical quantities                
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
            self.data['T'].fill(T0)
            
    def set_x(self, state = None, Z = None, perturb = 0):
        """
        Set initial ionization state.  If Z is None, assume constant ion fraction 
        of 1 / (1 + Z) for all elements.  Can be overrideen by 'state', which can be
        'equilibrium', and maybe eventually other options (e.g. perturbed out of
        equilibrium slightly perhaps).
        """       
        
        if state == 'equilibrium':
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
                        self.data[name] = np.ones(self.dims)
                    else:
                        self.data[name] = tiny_number * np.ones(self.dims)
        
        else:
            for species in self.ion_species:
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
            self.data['rho'].fill(rho0)   
            
        # Set hydrogen number density (which normalizes all other species)
        X = 0
        for i in xrange(len(self.abundances_by_number) - 1):
            name = util.z2element(i + 1)
            if not name.strip():
                continue
                
            ele = ELEMENT(name)
            X += self.abundances_by_number[i] * ele.mass
                                                          
        self.n_H = self.data['rho'] / g_per_amu / X

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

    def InitializeFields(self):
        """
        Return dictionary of all fields.
        """
 
        fields = {}
        for field in FieldList:
            fields[field] = np.array(eval("map(self.Initialize{0}, self.grid)".format(field)))
        
        if self.pf['OutputRates']:
            for i in xrange(3):
                for j in xrange(int(self.pf['NumberOfSources'])):
                    fields['PhotoIonizationRate%i_src%i' % (i, j)] = np.zeros_like(self.fgrid)
                    fields['PhotoHeatingRate%i_src%i' % (i, j)] = np.zeros_like(self.fgrid)
                    fields['SecondaryIonizationRate%i_src%i' % (i, j)] = np.zeros([len(self.fgrid), 3])    
                        
                    fields['InjectedLyAFlux%i_src%i' % (i, j)] = np.zeros_like(self.fgrid)
                    
                    if i == 0:
                        fields['ContinuumLyAFlux_src%i' % j] = np.zeros_like(self.fgrid)
                    
                fields['CollisionalIonizationRate%i' % i] = np.zeros_like(self.fgrid)
                fields['RadiativeRecombinationRate%i' % i] = np.zeros_like(self.fgrid)
                fields['CollisionalExcitationCoolingRate%i' % i] = np.zeros_like(self.fgrid)
                fields['CollisionalIonizationCoolingRate%i' % i] = np.zeros_like(self.fgrid)
                fields['RecombinationCoolingRate%i' % i] = np.zeros_like(self.fgrid)
                fields['CollisionalExcitationCoolingRate%i' % i] = np.zeros_like(self.fgrid)
                                    
                if i == 2:
                    fields['DielectricRecombinationRate'] = np.zeros_like(self.fgrid)
                    fields['DielectricRecombinationCoolingRate'] = np.zeros_like(self.fgrid)
        
        # Additional fields
        fields['dtPhoton'] = np.ones_like(self.fgrid)
        fields['OpticalDepth'] = np.zeros([len(self.fgrid), 3])        
        fields['Radius'] = self.r
        fields['ShellThickness'] = self.dx  
        fields['PhotonPackages'] = np.zeros(3)
        fields['HubbleCoolingRate'] = np.ones_like(self.fgrid)
        fields['SpinTemperature'] = np.ones_like(self.fgrid)
        fields['BrightnessTemperature'] = np.ones_like(self.fgrid)
                
        return fields                
                        
    def InitializeDensity(self, cell):
        """
        Initialize the gas density - depends on parameter DensityProfile as follows:
        
            DensityProfile:
                0: Uniform density given by InitialDensity parameter.
                1: Uniform density given by cosmic mean at z = InitialRedshift.
        """        
                
        if self.pf['CosmologicalExpansion'] == 0: 
            density = self.pf['DensityUnits']
        elif self.pf['CosmologicalExpansion'] == 1: 
            density = self.cosm.MeanBaryonDensity(self.pf['InitialRedshift'])
        
        if self.pf['Clump']: 
            if self.pf['ClumpDensityProfile'] == 0:
                if (cell >= self.pf['GridDimensions'] * (self.pf['ClumpPosition'] - self.pf['ClumpRadius'])) and \
                   (cell <= self.pf['GridDimensions'] * (self.pf['ClumpPosition'] + self.pf['ClumpRadius'])):
                    density *= self.pf['ClumpOverdensity']
            if self.pf['ClumpDensityProfile'] == 1:
                density += density * self.pf['ClumpOverdensity'] * \
                    np.exp(-(cell - self.pf['ClumpPosition'] * pf['GridDimensions'])**2 / 2. / self.pf['ClumpRadius']**2)
                        
        return density
        
    def InitializeTemperature(self, cell):
        """
        Initialize temperature - depends on parameter TemperatureProfile as follows:
        
            TemperatureProfile:
                0: Uniform temperature given by InitialTemperature
                1: Uniform temperature assuming Tk = Tcmb before decoupling, and
                   Tk ~ (1 + z)^2 after decoupling.
        """
                
        if self.pf['CosmologicalExpansion'] == 0: 
            temperature = self.pf['InitialTemperature']    
        elif self.pf['CosmologicalExpansion'] == 1: 
            temperature = self.cosm.Tgas(self.pf['InitialRedshift'])
                
        if self.pf['Clump']:
            if (cell >= self.pf['GridDimensions'] * (self.pf['ClumpPosition'] - self.pf['ClumpRadius'])) and \
               (cell <= self.pf['GridDimensions'] * (self.pf['ClumpPosition'] + self.pf['ClumpRadius'])):
                temperature = self.pf['ClumpTemperature']
        
        return temperature
        
    def InitializeIonization(self, cell):
        """
        Initialize ionization state - depends on parameter IonizationProfile as follows:
        
            IonizationProfile:
                0: Uniform ionization state given by InitialHIIFraction
                1: Gas within 'StartRadius' has x_i = 0.9999, InitialHIIFraction elsewhere
                   
        Returns the HII fraction in 'cell'.
        """
                
        if self.pf['IonizationProfile'] == 0: 
            ionization = self.pf['InitialHIIFraction']
        
        return ionization    
        

        