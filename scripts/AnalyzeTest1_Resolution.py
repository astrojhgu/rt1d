"""
AnalyzeTest1_Resolution.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2011-07-02.

Description: Analyze results of RT06 test problem #1 run with differing spatial resolutions.

Notes: Supply parameter file as commmand line argument.     
     
"""

import misc
import pylab as pl

r, x1, x2, x3, x4 = misc.readtab('c100/RT_Test1_RadialProfiles.dat')
pl.semilogy(r, x1, ls = '-', color = 'k', label = r'$n_{\mathrm{c}} = 100$')
pl.semilogy(r, x3, ls = '-', color = 'k')
pl.semilogy(r, x4, ls = '-', color = 'k')
r, x1, x2, x3, x4 = misc.readtab('c200/RT_Test1_RadialProfiles.dat')
pl.semilogy(r, x1, ls = '--', color = 'blue', label = r'$n_{\mathrm{c}} = 200$')
pl.semilogy(r, x3, ls = '--', color = 'blue')
pl.semilogy(r, x4, ls = '--', color = 'blue')
r, x1, x2, x3, x4 = misc.readtab('c400/RT_Test1_RadialProfiles.dat')
pl.semilogy(r, x1, ls = ':', color = 'red', label = r'$n_{\mathrm{c}} = 400$')
pl.semilogy(r, x3, ls = ':', color = 'red')
pl.semilogy(r, x4, ls = ':', color = 'red')
r, x1, x2, x3, x4 = misc.readtab('c800/RT_Test1_RadialProfiles.dat')
pl.semilogy(r, x1, ls = '-.', color = 'green', label = r'$n_{\mathrm{c}} = 800$')
pl.semilogy(r, x3, ls = '-.', color = 'green')
pl.semilogy(r, x4, ls = '-.', color = 'green')
r, x1, x2, x3, x4 = misc.readtab('c1600/RT_Test1_RadialProfiles.dat')
pl.semilogy(r, x1, ls = '-', color = 'magenta', label = r'$n_{\mathrm{c}} = 1600$')
pl.semilogy(r, x3, ls = '-', color = 'magenta')
pl.semilogy(r, x4, ls = '-', color = 'magenta')
r, x1, x2, x3, x4 = misc.readtab('c3200/RT_Test1_RadialProfiles.dat')
pl.semilogy(r, x1, ls = '--', color = 'cyan', label = r'$n_{\mathrm{c}} = 3200$')
pl.semilogy(r, x3, ls = '--', color = 'cyan')
pl.semilogy(r, x4, ls = '--', color = 'cyan')
pl.ylim(1e-5, 1.5)
pl.xlim(0, 1.01)
pl.xlabel(r'$r / L_{\mathrm{box}}$')
pl.ylabel(r'$x_{\mathrm{HI}}$')
pl.legend(loc = 'lower right', frameon = False)
pl.savefig('RadialProfile_Resolution.png')
pl.clf()

trec, rdiff = misc.readtab('c100/RT_Test1_IfrontEvolution.dat')
pl.plot(trec, rdiff, ls = '-', color = 'k', label = r'$n_{\mathrm{c}} = 100$')
trec, rdiff = misc.readtab('c200/RT_Test1_IfrontEvolution.dat')
pl.plot(trec, rdiff, ls = '--', color = 'blue', label = r'$n_{\mathrm{c}} = 200$')
trec, rdiff = misc.readtab('c400/RT_Test1_IfrontEvolution.dat')
pl.plot(trec, rdiff, ls = ':', color = 'red', label = r'$n_{\mathrm{c}} = 400$')
trec, rdiff = misc.readtab('c800/RT_Test1_IfrontEvolution.dat')
pl.plot(trec, rdiff, ls = '-.', color = 'green', label = r'$n_{\mathrm{c}} = 800$')
trec, rdiff = misc.readtab('c1600/RT_Test1_IfrontEvolution.dat')
pl.plot(trec, rdiff, ls = '-', color = 'magenta', label = r'$n_{\mathrm{c}} = 1600$')
trec, rdiff = misc.readtab('c3200/RT_Test1_IfrontEvolution.dat')
pl.plot(trec, rdiff, ls = '--', color = 'cyan', label = r'$n_{\mathrm{c}} = 3200$')
pl.ylim(0.9, 1.05)
pl.xlabel(r'$t / t_{\mathrm{rec}}$')
pl.ylabel(r'$r / r_{\mathrm{anl}}$')
pl.legend(loc = 'lower right', frameon = False)
pl.savefig('IfrontEvolution_Resolution.png')
pl.clf()