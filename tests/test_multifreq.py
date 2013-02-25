"""

test_multifreq.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sun Jan 20 14:58:19 2013

Description: Compare results using continuous and discrete emission on
RT06 problems 0.3 and 2, for now. 

"""

import rt1d
import pylab as pl

# RT06 0.3 - continuous, discrete, and optically_thin = 1
sim_c = rt1d.run.RT(pf = {'problem_type': 0})
sim_d = rt1d.run.RT(pf = {'problem_type': 0.1, 
    'dtDataDump': None, 'logdtDataDump': 0.25})
sim_t = rt1d.run.RT(pf = {'problem_type': 0, 'optically_thin': 1,
    'dtDataDump': None, 'logdtDataDump': 0.25})

anl_c = rt1d.analysis.Analyze(sim_c.checkpoints)
anl_d = rt1d.analysis.Analyze(sim_d.checkpoints)
anl_t = rt1d.analysis.Analyze(sim_t.checkpoints)

t1, z1, xHI1 = anl_c.CellTimeEvolution(field = 'h_1')
t1, z1, T1 = anl_c.CellTimeEvolution(field = 'T')

t2, z2, xHI2 = anl_d.CellTimeEvolution(field = 'h_1')
t2, z2, T2 = anl_d.CellTimeEvolution(field = 'T')

t3, z3, xHI3 = anl_t.CellTimeEvolution(field = 'h_1')
t3, z3, T3 = anl_t.CellTimeEvolution(field = 'T')

mp = rt1d.analysis.multiplot(dims = (2, 1), share_all = False, 
    useAxesGrid = False, panel_size = (0.5, 1))
    
s_per_yr = rt1d.physics.Constants.s_per_yr
mp.grid[0].loglog(t1 / s_per_yr, xHI1, color = 'k')
mp.grid[1].loglog(t1 / s_per_yr, T1, color = 'k')
mp.grid[0].scatter(t2 / s_per_yr, xHI2, color = 'b', marker = '+', s = 40,
    label = r'$n_{\nu} = 4$')
mp.grid[1].scatter(t2 / s_per_yr, T2, color = 'b', marker = '+', s = 40)
mp.grid[0].scatter(t3 / s_per_yr, xHI3, color = 'g', marker = 'o',
    facecolors = 'none', s = 40, label = r'$\tau \ll 1$')
mp.grid[1].scatter(t3 / s_per_yr, T3, color = 'g', marker = 'o',
    facecolors = 'none', s = 40)

mp.grid[0].set_xlim(1e-6, 1e7)
mp.grid[1].set_xlim(1e-6, 1e7)
mp.grid[0].set_ylim(1e-8, 1.5)
mp.grid[1].set_ylim(1e2, 1e5)
    
mp.grid[0].set_ylabel(r'$x_{\mathrm{HI}}$')
mp.grid[1].set_ylabel(r'$T \ (\mathrm{K})$')
mp.grid[1].set_xlabel(r'$t \ (\mathrm{yr})$')
mp.grid[0].legend(loc = 'lower left', frameon = False)
mp.fix_ticks()
pl.savefig('test_multifreq_sz.png')
raw_input('<enter> for HII region test')
pl.close()

# RT06 2 - continuous, discrete
sim_c = rt1d.run.RT(pf = {'problem_type': 2})
sim_d = rt1d.run.RT(pf = {'problem_type': 2.1})

anl_c = rt1d.analysis.Analyze(sim_c.checkpoints)
anl_d = rt1d.analysis.Analyze(sim_d.checkpoints)

ax = anl_c.TemperatureProfile()
anl_d.TemperatureProfile(ax = ax, color = 'b')

raw_input('<enter> for radial profiles of xHI & xHII')

ax = anl_c.IonizationProfile()
anl_d.IonizationProfile(ax = ax, color = 'b')
pl.savefig('test_multifreq_HII.png')
raw_input('')
