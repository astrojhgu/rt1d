"""

ProblemTypes.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Mar  7 15:53:15 2012

Description: Non-default parameter sets for certain test problems.

Note: Integer problem types imply use of a continuous SED, while their
non-integer counterparts imply a discrete SED, except for ProblemType = 1 = 1.1.
I generally (for clarity) order parameters from top to bottom in the following
way:
    Units, time/data dump interval
    Integral tabulation
    Initial conditions
    Source parameters
    Physics parameters
    
More notes:
-Non-integer problem types are the same as their integer counterparts
 but with discrete SEDs rather than continuous ones.
-A problem type > 10 (or < -10) corresponds to the same problem as 
 problem_type % 10, except helium is included.
-To compare apples to apples, remember to modify density_units so that the
 absolute number density of hydrogen remains unchanged. Or, modify
 the abundances parameter.

"""

import numpy as np
from ..physics.Constants import m_H, cm_per_kpc, cm_per_mpc, s_per_myr

def ProblemType(ptype):
    """
    Storage bin for predefined problem types, like those used in the 
    radiative transfer comparison project ('RT06').
    """
    
    ptype_int = int(ptype)
    if abs(ptype_int) > 10:
        ptype_int -= 10 * np.sign(ptype_int)
        
    ptype_mod1 = round(ptype - ptype_int, 1)
        
    # Single-zone, cosmological expansion test         
    if ptype_int == -1:
        pf = {
              "problem_type": 0.1, 
              "radiative_transfer": 0,
              "isothermal": 0,
              "expansion": 1,
              "grid_cells": 1,
              "length_units": 1e-4*cm_per_kpc, # 100 milliparsecs 
              "start_radius": 0.99, # cell = 1 milliparsec across
              "dtDataDump": 20, 
              "dzDataDump": 1,
              "initial_redshift": 500.,
              "final_redshift": 6,
              "restricted_timestep": ['ions', 'hubble'],
              "epsilon_dt": 0.01
             }
    
    # RT06-0.3, Single zone ionization/heating, then source switches off.
    if ptype_int == 0:
        pf = {
              "problem_type": 0,
              "plane_parallel": 1,
              "isothermal": 0,
              "density_units": m_H,
              "length_units": 1e-4 * cm_per_kpc, # 100 milliparsecs 
              "time_units": s_per_myr,
              "start_radius": 0.99,   # cell = 1 milliparsec across
              "grid_cells": 1, 
              
              "stop_time": 10, 
              "logdtDataDump": 0.1,
              "dtDataDump": None, 
              "initial_timestep": 1e-15,
              "max_timestep": 0.1,
              "restricted_timestep": ['ions', 'electrons', 'energy'],
                           
              "initial_temperature": 1e2,
              "initial_ionization": [1e-8],
              
              "source_type": 1,
              "source_qdot": 1e12,
              "source_lifetime": 0.5,
              
              "spectrum_type": 1,
              "tau_ifront": [0],
              
              "spectrum_Emin": 13.6,
              "spectrum_Emax": 100.,
              "spectrum_EminNorm": 0.1,
              "spectrum_EmaxNorm": 100.,
              "spectrum_smallest_x": 1e-10,
                           
             }
             
    # RT06-1, RT1: Pure hydrogen, isothermal HII region expansion, 
    # monochromatic spectrum at 13.6 eV
    if ptype_int == 1:
        pf = {
              "problem_type": 1, 
              "density_units": 1e-3 * m_H,
              "length_units": 6.6 * cm_per_kpc, 
              "stop_time": 500.0, 
              "isothermal": 1,
              "species": [1],
              "initial_temperature": 1e4,
              "initial_ionization": [1.2e-3], 
              "source_type": 0, 
              "source_qdot": 5e48, 
              "spectrum_E": [13.6],
              "spectrum_LE": [1.0],
             }
            
    # RT06-2: Pure hydrogen, HII region expansion, temperature evolution 
    # allowed, *continuous spectrum*
    if ptype_int == 2:
        pf = {
              "problem_type": 2, 
              "density_units": 1e-3 * m_H,
              "length_units": 6.6 * cm_per_kpc, 
              "stop_time": 500.0, 
              "isothermal": 0,
              "restricted_timestep": ['ions', 'energy'],
              "species": [1],
              "secondary_ionization": 0,
              "initial_temperature": 1e2,
              "initial_ionization": [1.2e-3], 
              "source_type": 1, 
              "source_temperature": 1e5,
              "spectrum_type": 1,
              "source_qdot": 5e48,
             }
        
    # RT06-3: I-front trapping in a dense clump and the formation of a shadow,
    # continuous blackbody spectrum
    if ptype_int == 3:
        pf = {
              "problem_type": 3,  
              "plane_parallel": 1,
              "density_units": 2e-4 * m_H,
              "length_units": 6.6 * cm_per_kpc,

              "stop_time": 15.0, 
              "dtDataDump": 1.0,
              "isothermal": 0,  
              "initial_temperature": 8e3,
              "initial_ionization": [1e-6],
              "source_type": 1, 
              "source_qdot": 1e6,
              "spectrum_type": 1,
              
              "restricted_timestep": ['ions', 'electrons', 'energy'],
              
              "spectrum_Emin": 13.6,
              "spectrum_Emax": 100.,
              "spectrum_EminNorm": 0.1,
              "spectrum_EmaxNorm": 100.,
              
              "clump": 1,
              "clump_position": 5.0 / 6.6,
              "clump_overdensity": 200.,
              "clump_radius": 0.8 / 6.6,
              "clump_temperature": 40.,
              "clump_profile": 0,
              "clump_ionization": 1e-6,

             }
             
    if ptype_mod1 != 0:
        pf.update({'photon_conserving': 1})
        
        # Change discrete spectrum: 0.1 = Mirocha et al. 2012
        #                           0.2 = Wise & Abel 2011
        if ptype_mod1 == 0.1:
            pf.update({'spectrum_E': [17.98, 31.15, 49.09, 76.98]})
            pf.update({'spectrum_LE': [0.23, 0.36, 0.24, 0.06]})
        if ptype_mod1 == 0.2:
            pf.update({'spectrum_E': [18.29, 31.46, 49.13, 77.23]})
            pf.update({'spectrum_LE': [0.24, 0.35, 0.23, 0.06]})
             
    if ptype >= 10:
        pf.update({'species': [1, 2], 'abundances': [1.0, 0.08],
            'initial_ionization':[pf['initial_ionization']]*2})
                     
    return pf    