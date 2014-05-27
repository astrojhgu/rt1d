"""

test_cxrb.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Oct  2 17:10:22 MDT 2013

Description: Use ParameterizedSource to mimic flux of CXRB photons, using
clump test problem.

"""

import rt1d, glorb
import numpy as np
import pylab as pl
from multiplot import multipanel
from rt1d.physics.Constants import *

# Initialize radiation background
src_pars = \
{
 'sfrd': lambda z: 0.01 / (1. + z)**3.,
 'source_type': 'bh',
 'model': -1,
 'spectrum_type': 'pl',
 'spectrum_alpha': -1.5,
 'spectrum_Emin': 2e2,
 'spectrum_Emax': 3e4,
 'spectrum_EminNorm': 5e2,
 'spectrum_EmaxNorm': 8e3,
 'approx_xray': 0,
 'approx_helium': 0,
 'frequency_bins': 64,
 'hybrid_integrator': True,
 'resolve_tau_by': 15,
}

rad = glorb.evolve.RadiationBackground(**src_pars)

flux_ion = np.zeros_like(rad.igm.E)
flux_neu = np.zeros_like(rad.igm.E)
for i, E in enumerate(rad.igm.E): 
    flux_ion[i] = rad.AngleAveragedFlux(6., E, zf=10, energy_units=True,
        xavg=lambda z: 1.0)
    flux_neu[i] = rad.AngleAveragedFlux(6., E, zf=10, energy_units=True,
        xavg=lambda z: 0.0)

pl.loglog(rad.igm.E, flux_neu, color='k')
pl.loglog(rad.igm.E, flux_ion, color='k', ls='--')
pl.xlabel(r'$h\nu \ (\mathrm{eV})$')
pl.ylabel(r'$J_{\nu} \ (\mathrm{erg} \ \mathrm{s}^{-1} \ \mathrm{cm}^{-2} \ \mathrm{Hz}^{-1} \ \mathrm{sr}^{-1})$')

raw_input('<enter> for simulations')

# Should normalize to same E < 1.5 keV energy

flux_i = lambda E: np.interp(E, rad.igm.E, flux_ion) * 4. * np.pi / ev_per_hz
flux_n = lambda E: np.interp(E, rad.igm.E, flux_neu) * 4. * np.pi / ev_per_hz

sim_ion = rt1d.run.Simulation(problem_type=3, grid_cells=256,
    tables_dlogN=[0.05], source_type='parameterized',
    source_Lbol=np.trapz(map(flux_i, rad.igm.E)),
    spectrum_function=lambda E: flux_i(E),
    density_units=rad.cosm.MeanBaryonDensity(20.),
    initial_temperature=rad.cosm.Tgas(20.),
    clump_temperature=rad.cosm.Tgas(20.) / 2.,
    spectrum_Emin=rad.igm.E0, 
    spectrum_Emax=rad.igm.E1,
    secondary_ionization=1)
sim_ion.run()

sim_neu = rt1d.run.Simulation(problem_type=3, grid_cells=256,
    tables_dlogN=[0.05], source_type='parameterized',
    source_Lbol=np.trapz(map(flux_n, rad.igm.E)),
    spectrum_function=lambda E: flux_n(E),
    density_units=rad.cosm.MeanBaryonDensity(20.),
    initial_temperature=rad.cosm.Tgas(20.),
    clump_temperature=rad.cosm.Tgas(20.) / 2.,
    source_normalized=True, spectrum_Emin=rad.igm.E0, 
    spectrum_Emax=rad.igm.E1,
    secondary_ionization=1)
sim_neu.run()

pl.close()

anl_i = rt1d.analyze.Simulation(sim_ion.checkpoints)
anl_n = rt1d.analyze.Simulation(sim_neu.checkpoints)

# Plot solutions at 1, 3, 5, 15 Myr
t = [1, 3, 5, 15]
ls = [':', '--', '-.', '-']

mp = multipanel(dims=(2, 1))
    
# Plot initial conditions
mp.grid[0].semilogy(anl_i.grid.r_mid / anl_i.grid.length_units, 
    anl_i.data['dd0000']['h_1'], color='g', ls='-')
mp.grid[1].semilogy(anl_i.grid.r_mid / anl_i.grid.length_units, 
    anl_i.data['dd0000']['Tk'], color='g', ls='-')
    
ct = 0
for dd in anl_i.data.keys():
    t_code = anl_i.data[dd]['time'] / anl_i.pf['time_units']
    
    if t_code not in t: 
        continue
        
    this_t = int(t_code)
    
    mp.grid[0].semilogy(anl_i.grid.r_mid / anl_i.grid.length_units, anl_i.data[dd]['h_1'], 
        color='k', ls=ls[ct], 
        label = r'$t = %i \ \mathrm{Myr}$' % this_t)
    mp.grid[1].semilogy(anl_i.grid.r_mid / anl_i.grid.length_units, anl_i.data[dd]['Tk'], 
        color='k', ls=ls[ct])
    
    # neutral IGM case
    mp.grid[0].semilogy(anl_n.grid.r_mid / anl_n.grid.length_units, anl_n.data[dd]['h_1'], 
        color='b', ls=ls[ct])
    mp.grid[1].semilogy(anl_n.grid.r_mid / anl_n.grid.length_units, anl_n.data[dd]['Tk'], 
        color='b', ls=ls[ct])    
    
    ct += 1

mp.grid[0].set_ylim(1e-3, 1.5)
mp.grid[1].set_ylim(1, 8e4)
                                        
for i in xrange(2):
    mp.grid[i].set_xlim(0.0, 1.0)
                            
mp.grid[0].set_xlabel(r'$x / L_{\mathrm{box}}$')    
mp.grid[0].set_ylabel('Neutral Fraction')
mp.grid[1].set_ylabel(r'Temperature $(K)$')    
mp.fix_ticks()

mp.grid[0].legend(loc='lower right', frameon=False)
                
pl.draw() 

