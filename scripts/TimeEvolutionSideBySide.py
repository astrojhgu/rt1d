"""

tdep_comparison.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sat Jun  9 09:13:11 2012

Description: 

"""

import os
import rt1d.analysis as rta
from jmpy.multiplot import *

const = rta.Analyze('sed_const.dat', retabulate = False)
var = rta.Analyze('sed_variable.dat', retabulate = False)

if not os.path.exists('frames'):
    os.mkdir('frames')

mp = multiplot(dims = (1, 3), panel_size = (2, 0.85), 
    useAxesGrid = False, share_all = False, padding = (0.3, 0))

Ndd = int(const.pf['StopTime'] / const.pf['dtDataDump'])

# 100 data dumps
for i in xrange(Ndd + 1):
    
    const.rs.PlotSpectrum(t = 0, color = 'k', mp = mp.grid[0])
    var.rs.PlotSpectrum(t = var.data[i].t, color = 'b', mp = mp.grid[0])
    
    mp.grid[1].semilogy(const.data[i].r / const.pf.LengthUnits, const.data[i].x_HI, 
        color = 'k', ls = '-', label = r'$x_{\mathrm{HI}}$')
    mp.grid[1].semilogy(const.data[i].r / const.pf.LengthUnits, const.data[i].x_HII, 
        color = 'k', ls = '--', label = r'$x_{\mathrm{HII}}$')
    mp.grid[1].semilogy(var.data[i].r / var.pf.LengthUnits, var.data[i].x_HI, 
        color = 'b', ls = '-')
    mp.grid[1].semilogy(var.data[i].r / var.pf.LengthUnits, var.data[i].x_HII, 
        color = 'b', ls = '--')    
    
    mp.grid[2].semilogy(const.data[i].r / const.pf.LengthUnits, const.data[i].T, 
        color = 'k', ls = '-')
    mp.grid[2].semilogy(var.data[i].r / var.pf.LengthUnits, var.data[i].T, 
        color = 'b', ls = '-')
    
    mp.grid[1].set_xlabel(r'$r / L_{\mathrm{box}}$')
    mp.grid[1].set_ylabel(r'Species Fraction')
    mp.grid[2].set_xlabel(r'$r / L_{\mathrm{box}}$')
    mp.grid[2].set_ylabel(r'$T \ (\mathrm{K})$')
    
    mp.grid[0].annotate(r'$I_{\nu}(0)$', xy = (20, 1e-5))
    mp.grid[0].annotate(r'$I_{\nu}(t)$', xy = (20, 6e-6), color = 'b')
    
    mp.grid[1].set_yscale('log')
    mp.grid[2].set_yscale('log')
    const.rs.ax.set_ylim(1e-6, 2e-3)
    mp.grid[1].set_ylim(1e-5, 1.5)
    mp.grid[2].set_ylim(90, 1e5)
    mp.grid[1].legend(loc = 'lower right', frameon = False)
    
    title = r'$t = {0}$ Myr'.format(i)
    mp.grid[1].set_title(title)
    
    pl.draw()
    
    pl.savefig('frames/dd%s_3panel.png' % str(i).zfill(4))
    
    for i in xrange(3):
        mp.grid[i].clear()
    
    
    
    
    
    




