"""
SetDefaultParameterValues.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-10-19.

Description: Defaults for all different kinds of parameters, now sorted
into groups.
     
"""

from numpy import inf
from ..physics.Constants import m_H, cm_per_kpc, s_per_myr

defaults = []
pgroups = ['Grid', 'Source', 'Spectrum', 'Physics', 
           'Cosmology', 'Control']
for grp in pgroups:
    defaults.append('%sParameters()' % grp)
    
def SetAllDefaults():
    pf = {'problem_type': 1}
    
    for pset in defaults:
        eval('pf.update(%s)' % pset)
        
    return pf
    
def GridParameters():
    pf = \
        {
        "grid_cells": 32,
        "start_radius": 0.01,
        
        "density_units": 1e-3 * m_H,
        "length_units": 6.6 * cm_per_kpc,  
        "time_units": s_per_myr,  
        
        "Z": [1],
        "abundance": [1.],
        "initial_ionization": [1.2e-3],
        "initial_temperature": 1e4,
        
        "approx_helium": 0,
        
        "clump": 0,  
        "clump_position": 0.1,  
        "clump_radius": 0.05,  
        "clump_overdensity": 100,  
        "clump_temperature": 100,    
        "clump_ionization": 1e-6,
        "clump_profile": 0,
                
        }

    return pf

def PhysicsParameters():
    pf = \
        {        
        "radiative_transfer": 1,
        "photon_conserving": 1, 
        "plane_parallel": 0,   
        "infinite_c": 1,  
        
        "secondary_ionization": 0,  
        "isothermal": 1,  
        "expansion": 0,  
        "compton_scattering": 0,
        "recombination": 'B', 
        
        "clumping_factor": 1,
        
        }
        
    return pf
    
def SourceParameters():
    pf = \
        {
        "source_type": 0,  
        
        "source_temperature": 1e5,  
        "source_qdot": 5e48,
        "source_mass": 1e3,  
        "source_fduty": 1,
        "source_tbirth": 0,
        "source_lifetime": 1e10,  
        "source_eta": 0.1,
        "source_isco": 6,  
        "source_rmax": 1e3,
        "source_cX": 1.0,
        
        "source_ion": 0,
        "source_ion2": 0,
        "source_heat": 0,
        "source_lya": 0,
        
        "source_table": None,
        
        }
        
    return pf
    
def SpectrumParameters():
    pf = \
        {        
        "spectrum_type": 0,
        "spectrum_evolving": False,
        
        "spectrum_fraction": 1,
        "spectrum_alpha": -1.5,
        "spectrum_Emin": 13.6,  
        "spectrum_Emax": 1e2,  
        "spectrum_EminNorm": None,
        "spectrum_EmaxNorm": None,
        
        "spectrum_linewidth": None,
        "spectrum_linecenter": None, 
         
        "spectrum_logN": 0,
        "spectrum_fcol": 1.7,
        "spectrum_file": None,
        "spectrum_pars": None,
        
        "spectrum_multigroup": 0,
        "spectrum_bands": None,
          
        "spectrum_t": None,  
        "spectrum_E": None,
        "spectrum_LE": None,  
                
        "spectrum_table": None,
        
        "spectrum_logNmin": None,
        "spectrum_logNmax": None,
        "spectrum_dlogN": [0.2],
        
        "spectrum_smallest_x": [1e-8],
        
        "spectrum_logxmin": [-4],
        "spectrum_dlogx": 0.1,
        "spectrum_dE": 5.,
        "spectrum_dt": s_per_myr,
        "spectrum_extrapolate": False,
        
        "spectrum_normed_by": 'energy',
        
        }
        
    return pf
    
def ControlParameters():
    pf = \
        {
        "epsilon_dt": 0.05,
        "dtDataDump": 1,
        "dzDataDump": None,
        'logdtDataDump': None,
        'logdzDataDump': None,
        "stop_time": 500,
        "initial_redshift": 20.,
        "final_redshift": 6.,
        "initial_timestep": 1e-8,
        "max_timestep": 1.,
        "restricted_timestep": ['ions'],
        
        "tau_ifront": 0.5,
        "optically_thin": 0,
        
        "parallelization": 0,
        "save_rate_coefficients": 1,
        
        "interp_method": 'cubic',
        }
        
    return pf
    
def CosmologyParameters():
    pf = \
        {
        "OmegaMatterNow": 0.272,
        "OmegaBaryonNow": 0.044,
        "OmegaLambdaNow": 0.728,
        "HubbleParameterNow": 0.702,
        "HeliumFractionByMass": 0.2477,
        "CMBTemperatureNow": 2.725,
        "HighRedshiftApprox": 0,
        "SigmaEight": 0.807
        }
        
    return pf    
    

