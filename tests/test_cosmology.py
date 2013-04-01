"""

test_cosmology.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Mar 28 16:34:18 2013

Description: 

"""

import rt1d
import pylab as pl
import numpy as np

sim = rt1d.run.Simulation(pf={'problem_type':-1,
    'initial_redshift': 1e3, 'initial_ionization': [0.049]})
sim.run()

anl = rt1d.analyze.Simulation(sim.checkpoints)

z = np.linspace(10, 1e3)
pl.loglog(z, sim.grid.cosm.TCMB(sim.grid.cosm.zdec) * \
    (1. + z)**2 / (1. + sim.grid.cosm.zdec)**2, color = 'k',
    label=r'analytic')
pl.loglog(z, sim.grid.cosm.TCMB(z), color = 'k', ls = ':')

z, T = anl.CellEvolution(field='T', redshift=True)
pl.loglog(z, T, color='b', label='rt1d')
z, Ts = anl.CellEvolution(field='Ts', redshift=True)
pl.loglog(z, Ts, color='b', ls = '--')

pl.xlabel(r'$z$')
pl.ylabel(r'$T_K$')
pl.xlim(10, sim.grid.zi)
pl.ylim(1, 2e3)

#try:
#    import glorb
#    ds = glorb.analysis.Analyze21cm(glorb.run.RadioBackground(**{'zi': 1100, 
#        'zfl': 6, 'dz': 1}))
#    pl.loglog(ds.data['z'], ds.data['Tk'], color='g', label='cosmorec')
#    pl.loglog(ds.data['z'], ds.data['Ts'], color='g', ls='--')    
#except:
#    pass
    
pl.legend(frameon=False, loc='lower right')    
raw_input('')    



