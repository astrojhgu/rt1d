"""

Simulate.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Jan  2 15:20:01 2013

Description: Run a simulation.

"""

import rt1d
from ..util.ReadParameterFile import *
from ..util import parse_kwargs, ReadParameterFile

class simulation:
    def __init__(self, pf, checkpoints, rt):
        self.pf = pf
        self.checkpoints = checkpoints
        self.data = checkpoints.data
        self.grid = checkpoints.grid
        self.rt = rt
        self.rs = rt.src
        self.rf = rt.rfield

def RTsim(pf = None):
    """
    Run a radiative transfer simulation from input parameter file, which
    can be a dictionary or a path to a text file.
    """
    
    if type(pf) is str:
        pf = ReadParameterFile(pf)
    elif type(pf) is dict:
        pf = parse_kwargs(**pf)
    else:
        pf = parse_kwargs()
    
    # Initialize grid object
    grid = rt1d.Grid(dims = pf['grid_cells'], length_units = pf['length_units'],
        start_radius = pf['start_radius'])
    
    # Set initial conditions
    grid.set_chem(Z = pf['species'], abundance = pf['abundances'], 
        isothermal = pf['isothermal'])
    grid.set_rho(rho0 = pf['density_units'])
    
    for i in xrange(len(pf['species'])):
        grid.set_x(Z = pf['species'][i], x = pf['initial_ionization'][i])       
    
    grid.set_T(pf['initial_temperature'])
    
    if pf['clump']:
        grid.make_clump(position = pf['clump_position'], radius = pf['clump_radius'], 
            temperature = pf['clump_temperature'], overdensity = pf['clump_overdensity'],
            ionization = pf['clump_ionization'], profile = pf['clump_profile'])
            
    # Initialize radiation source and radiative transfer solver
    rs = rt1d.sources.RadiationSourceIdealized(grid, **pf)
    rt = rt1d.Radiation(grid, rs, **pf)
    
    # To compute timestep
    timestep = rt1d.run.ComputeTimestep(grid, pf['epsilon_dt'])
    
    # For storing data
    checkpoints = rt1d.util.CheckPoints(pf = pf, grid = grid,
        dtDataDump = pf['dtDataDump'], time_units = pf['time_units'],
        stop_time = pf['stop_time'], initial_timestep = pf['initial_timestep'],
        logdtDataDump = pf['logdtDataDump'], source_lifetime = pf['source_lifetime'])
        
    if pf['initialize_only']:
        return simulation(pf, checkpoints, rt)   
            
    # Evolve chemistry + RT
    data = grid.data
    dt = pf['time_units'] * pf['initial_timestep']
    t = 0.0
    tf = pf['stop_time'] * pf['time_units']
    max_timestep = pf['time_units'] * pf['max_timestep']
    
    print '\nSolving radiative transfer...'
                
    dt_history = []
    pb = rt1d.run.ProgressBar(tf)
    while t < tf:
                                        
        # Evolve by dt
        data = rt.Evolve(data, t = t, dt = dt)
        t += dt 
        
        pb.update(t)
                        
        # Figure out next dt based on max allowed change in evolving fields
        new_dt = timestep.Limit(rt.chem.q_grid, rt.chem.dqdt_grid, 
            rt.rfield.tau_tot, tau_ifront = pf['tau_ifront'], 
            method = pf['restricted_timestep'])
        
        # Limit timestep further based on next DD and max allowed increase
        dt = min(new_dt, 2 * dt)
        dt = min(checkpoints.update(data, t, dt), max_timestep)
                
        # Save timestep history
        dt_history.append((t, dt))
        
        if pf['save_rate_coefficients']:
            checkpoints.store_kwargs(t, rt.kwargs)
            
        # Raise error if any funny stuff happens
        if dt < 0: 
            raise ValueError('ERROR: dt < 0.') 
        elif dt == 0:
            raise ValueError('ERROR: dt = 0.')  
        elif np.isnan(dt):  
            raise ValueError('ERROR: dt -> inf.')        
            
    pb.finish()
        
    sim = simulation(pf, checkpoints, rt)
    sim.dt_history = dt_history    
        
    return sim