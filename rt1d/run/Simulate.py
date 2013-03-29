"""

Simulate.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Jan  2 15:20:01 2013

Description: Run a simulation.

"""

import rt1d
import numpy as np
from ..util import parse_kwargs, ReadParameterFile
    
class Simulation:
    """
    Run a radiative transfer simulation from input parameter file, which
    can be a dictionary or a path to a text file.
    """
    def __init__(self, pf=None, init_grid=True, init_rs=True, init_tabs=True):
        if type(pf) is str:
            pf = ReadParameterFile(pf)
        elif type(pf) is dict:
            pf = parse_kwargs(**pf)
        else:
            pf = parse_kwargs(**{'problem_type': 1})
            
        self.pf = pf
        
        # Initialize grid object
        if init_grid:
            grid = rt1d.Grid(dims = pf['grid_cells'], 
                length_units = pf['length_units'], 
                start_radius = pf['start_radius'])
            
            # Set initial conditions
            if pf['expansion']:
                grid.set_cosmology(zi=pf['initial_redshift'],
                    compton_scattering=pf['compton_scattering'],
                    xi = pf['initial_ionization'][0])
            else:
                grid.set_chem(Z = pf['species'], abundance = pf['abundances'], 
                    isothermal = pf['isothermal'])
                grid.set_rho(rho0 = pf['density_units'])
                
                for i in xrange(len(pf['species'])):
                    grid.set_x(Z = pf['species'][i], 
                        x = pf['initial_ionization'][i])       
                
                grid.set_T(pf['initial_temperature'])
                
                if pf['clump']:
                    grid.make_clump(position=pf['clump_position'], 
                        radius=pf['clump_radius'], 
                        temperature=pf['clump_temperature'], 
                        overdensity=pf['clump_overdensity'],
                        ionization=pf['clump_ionization'], 
                        profile=pf['clump_profile'])
                    
            # To compute timestep
            self.timestep = rt1d.run.ComputeTimestep(grid, pf['epsilon_dt'])
            
            # For storing data
            self.checkpoints = rt1d.util.CheckPoints(pf=pf, grid=grid,
                dtDataDump=pf['dtDataDump'], time_units=pf['time_units'],
                stop_time=pf['stop_time'], 
                initial_timestep=pf['initial_timestep'],
                logdtDataDump=pf['logdtDataDump'], 
                source_lifetime=pf['source_lifetime'])
            
            self.grid = grid
            
        # Initialize radiation source and radiative transfer solver    
        if init_rs:     
            if self.pf['radiative_transfer']:
                self.rs = rt1d.sources.RadiationSources(grid, 
                    init_tabs=init_tabs, **pf)
                allsrcs = self.rs.all_sources    
            else:
                allsrcs = None
                
            self.rt = rt1d.Radiation(self.grid, allsrcs, **self.pf)

    def run(self):
        self.__call__() 
     
    def __call__(self):
        """ Evolve chemistry and radiative transfer. """
        
        data = self.grid.data.copy()
        dt = self.pf['time_units'] * self.pf['initial_timestep']
        t = 0.0
        tf = self.pf['stop_time'] * self.pf['time_units']
        max_timestep = self.pf['time_units'] * self.pf['max_timestep']
        
        print '\nSolving radiative transfer...'
                    
        dt_history = []
        pb = rt1d.run.ProgressBar(tf)
        while t < tf:
                    
            z = None                 
            if self.grid.expansion:
                z = self.grid.cosm.TimeToRedshiftConverter(0, t, self.grid.zi)                         
                                            
            # Evolve by dt
            data = self.rt.Evolve(data, t = t, dt = dt)
            t += dt 
            
            tau_tot = None
            if hasattr(self.rt, 'rfield'):
                tau_tot = self.rt.rfield.tau_tot
                            
            # Figure out next dt based on max allowed change in evolving fields
            new_dt = self.timestep.Limit(self.rt.chem.q_grid, 
                self.rt.chem.dqdt_grid, z=z, tau=tau_tot, 
                tau_ifront=self.pf['tau_ifront'], 
                method=self.pf['restricted_timestep'])
            
            # Limit timestep further based on next DD and max allowed increase
            dt = min(new_dt, 2 * dt)
            dt = min(self.checkpoints.update(data, t, dt), max_timestep)
                    
            # Save timestep history
            dt_history.append((t, dt))
            
            if self.pf['save_rate_coefficients']:
                self.checkpoints.store_kwargs(t, self.rt.kwargs)
                
            # Raise error if any funny stuff happens
            if dt < 0: 
                raise ValueError('ERROR: dt < 0.') 
            elif dt == 0:
                raise ValueError('ERROR: dt = 0.')  
            elif np.isnan(dt):  
                raise ValueError('ERROR: dt -> inf.')      
                
            pb.update(t)
                
        pb.finish()
            
            