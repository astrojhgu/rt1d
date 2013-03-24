"""

test_optimization_n4.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Feb 25 09:27:17 2013

Description: Find optimal 4-bin 10^5 BB SED.

"""

import rt1d, sys
import numpy as np
import pylab as pl

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1    

erg_per_ev = rt1d.physics.Constants.erg_per_ev 

### INPUT
burn = 1e5
steps = 1e5
mcmc = False
###

# Set chemistry - hydrogen only for this test
Z = [1]

# Set column density interval of interest
logN = [np.linspace(16, 20, 41)]

# Set source properties - grab radiation source for RT06 #2 (10^5 K BB)
src = {'problem_type': 2}

# Compute posterior probability via MCMC
sedop = rt1d.run.Optimization(logN=logN, Z=Z, nfreq=4, 
    rs=src, mcmc=mcmc, isothermal=False, thinlimit=False)
step = [25.] * sedop.nfreq
step.extend([0.25] * sedop.nfreq)
#guesses = [17.98, 31.15, 49.09, 76.98, 0.23, 0.36, 0.24, 0.06]
limits = [(13.6, 100.)] * 4
limits.extend([(0., 1.)] * 4)
sedop(steps, burn=burn, limits=limits, err=0.2, afreq=30, gamma=0.99)

# Plot optimal Phi and Psi functions vs. HI column density
pars = sedop.sampler.xarr_mode
Eopt, LEopt = np.array(pars[:len(pars) / 2]), np.array(pars[len(pars) / 2:])

if rank > 0:
    sys.exit()

print Eopt, LEopt, np.sum(LEopt)
#print sedop.sampler.facc_b, sedop.sampler.facc

best_phi = sedop.discrete_tabs(Eopt, LEopt)['logPhi_h_1']
best_psi = sedop.discrete_tabs(Eopt, LEopt)['logPsi_h_1']

mp = rt1d.analyze.multiplot(dims = (2, 1), useAxesGrid = False)

mp.grid[0].loglog(10**sedop.logN[0], 10**sedop.rs.tabs['logPhi_h_1'], color = 'k')
mp.grid[0].loglog(10**sedop.logN[0], 10**best_phi, color = 'b')
mp.grid[0].set_ylim(1e6, 1e11)
mp.grid[0].set_ylabel(r'$\Phi$')

mp.grid[1].loglog(10**sedop.logN[0], 10**sedop.rs.tabs['logPsi_h_1'], color = 'k')
mp.grid[1].loglog(10**sedop.logN[0], 10**best_psi, color = 'b')
mp.grid[1].set_ylim(1e-4, 2)
mp.grid[1].set_xlabel(r'$N \ (\mathrm{cm}^{-2})$')
mp.grid[1].set_ylabel(r'$\Psi$')
mp.fix_ticks()

pl.draw()
pl.savefig('phi_psi.png')
raw_input('')
pl.close()

mp = rt1d.analyze.multiplot(dims = (2, 2), useAxesGrid = False)

for i in xrange(4):
    bins = np.linspace(Eopt[i] - 10, Eopt[i] + 10, 41)
    mp.grid[i].hist(sedop.sampler.chain[...,i], bins=bins)
    mp.grid[i].plot([Eopt[i]] * 2, mp.grid[i].get_ylim(), color = 'k', ls = ':')
    
mp.fix_ticks()
pl.draw()
pl.savefig('CIs.png')
raw_input('')    
pl.close()
