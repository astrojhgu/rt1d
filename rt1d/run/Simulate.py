"""

Simulate.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Jan  2 15:20:01 2013

Description: Run a simulation.

"""

import rt1d
import numpy as np
from ..physics.Constants import s_per_myr
from ..util import parse_kwargs, ReadParameterFile, ProgressBar
    
class Simulation:
    """
    Run a radiative transfer simulation from input parameter file (which
    can be a dictionary or a path to a text file) and/or kwargs.
    """
    def __init__(self, pf=None, init_grid=True, init_rs=True, init_tabs=True, 
        **kwargs):
        if pf is not None:
            if type(pf) is str:
                pf = ReadParameterFile(pf)
            elif type(pf) is dict:
                pf = parse_kwargs(**pf)
        else:
            pf = {}
            if kwargs:
                pf.update(parse_kwargs(**kwargs))
            else:
                pf.update(parse_kwargs(**{'problem_type': 1}))
                        
        self.pf = pf
        
        # Initialize grid object
        if init_grid:
            grid = rt1d.static.Grid(dims=pf['grid_cells'], 
                length_units=pf['length_units'], 
                start_radius=pf['start_radius'])
            
            grid.set_physics(isothermal=pf['isothermal'], 
                compton_scattering=pf['compton_scattering'],
                secondary_ionization=pf['secondary_ionization'], 
                expansion=pf['expansion'], 
                recombination=pf['recombination'])
            
            # Set initial conditions
            if pf['expansion']:
                grid.set_cosmology(initial_redshift=pf['initial_redshift'],
                    OmegaMatterNow=pf['OmegaMatterNow'], 
                    OmegaLambdaNow=pf['OmegaLambdaNow'], 
                    OmegaBaryonNow=pf['OmegaBaryonNow'],
                    HubbleParameterNow=pf['HubbleParameterNow'],
                    HeliumFractionByMass=pf['HeliumFractionByMass'], 
                    CMBTemperatureNow=pf['CMBTemperatureNow'],
                    approx_highz=pf['approx_highz'])    
                grid.set_chemistry(Z=pf['Z'], abundance=pf['abundance'],
                    approx_helium=pf['approx_helium'])
                grid.set_density(grid.cosm.rho_b_z0 \
                    * (1. + pf['initial_redshift'])**3)
                grid.set_temperature(grid.cosm.TCMB(pf['initial_redshift']))
                
                for i, Z in enumerate(pf['Z']):
                    grid.set_ionization(Z=Z, x=pf['initial_ionization'][i])
                
                grid.data['n'] = grid.particle_density(grid.data, 
                    z=pf['initial_redshift'])
                    
            else:
                grid.set_chemistry(Z=pf['Z'], abundance=pf['abundance'],
                    approx_helium=pf['approx_helium'])
                grid.set_density(pf['density_units'])
                
                for i, Z in enumerate(pf['Z']):
                    grid.set_ionization(Z=Z, x=pf['initial_ionization'][i])
                
                grid.set_temperature(pf['initial_temperature'])
                grid.data['n'] = grid.particle_density(grid.data)
                
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
                dtDataDump=pf['dtDataDump'], 
                dzDataDump=pf['dzDataDump'], 
                time_units=pf['time_units'],
                stop_time=pf['stop_time'], 
                final_redshift=pf['final_redshift'],
                initial_redshift=pf['initial_redshift'],
                initial_timestep=pf['initial_timestep'],
                logdtDataDump=pf['logdtDataDump'], 
                source_lifetime=pf['source_lifetime'])
            
            self.grid = grid
            
        # Initialize radiation source and radiative transfer solver    
        if init_rs:     
            if self.pf['radiative_transfer']:
                self.rs = rt1d.sources.CompositeSource(grid, 
                    init_tabs=init_tabs, **pf)
                allsrcs = self.rs.all_sources    
            else:
                allsrcs = None
                
            self.rt = rt1d.evolve.Radiation(self.grid, allsrcs, **self.pf)

    def run(self):
        self.__call__() 
     
    def __call__(self):
        """ Evolve chemistry and radiative transfer. """
        
        data = self.grid.data.copy()
        t = 0.0
        dt = self.pf['time_units'] * self.pf['initial_timestep']
        tf = self.pf['stop_time'] * self.pf['time_units']
        z = self.pf['initial_redshift']
        dz = self.pf['dzDataDump']
        zf = self.pf['final_redshift']
                
        max_timestep = self.pf['time_units'] * self.pf['max_timestep']
        
        if self.pf['radiative_transfer']:
            print '\nEvolving radiative transfer...'
        else:
            print '\nEvolving ion densities...'
                    
        dt_history = []
        pb = rt1d.util.ProgressBar(tf)
        pb.start()
        while t < tf:
                                         
            # Evolve by dt
            data = self.rt.Evolve(data, t=t, dt=dt, z=z)
            t += dt 
            
            if self.grid.expansion:
                z -= dt / self.grid.cosm.dtdz(z)
            
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
            dt = min(self.checkpoints.next_dt(t, dt), max_timestep)
            
            # Limit timestep based on next RD
            if self.checkpoints.redshift_dumps:
                dz = self.checkpoints.next_dz(z, dz)
                dt = min(dt, dz*self.grid.cosm.dtdz(z))
                                
            self.checkpoints.update(data, t, z)
                    
            # Save timestep history
            dt_history.append((t, dt))
            
            if self.pf['save_rate_coefficients']:
                self.checkpoints.store_kwargs(t, z, self.rt.kwargs)    
                
            pb.update(t)
            
            if z <= self.pf['final_redshift']:
                break
                
        pb.finish()
            
            