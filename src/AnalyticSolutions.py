"""
AnalyticSolutions.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-11-19.

Description: Analytic solutions to the propagation of an I-front in static and expanding universes.
     
"""

import numpy as np

cm_per_kpc = 3.08568 * 10**21
s_per_myr = 365.25 * 24 * 3600 * 10**6

def StaticSolution(t_max, Ndot, T, n_H):
    """
    Returns the position of an I-front as a function of time in a static universe.
    
    Enter t_max in Myr, Ndot in photons / sec, T in K, n_H in cm^-3.
    """
    
    alpha_HII = 2.6e-13 * (T / 1.e4)**-0.85
    trec = 1. / alpha_HII / n_H / s_per_myr
    rs = (3. * Ndot / 4. / np.pi / alpha_HII / n_H**2)**(1. / 3.) / cm_per_kpc
    
    func = lambda t: rs * (1. - np.exp(-t / trec))**(1. / 3.)
    t_anl = np.arange(0, t_max, 0.05)
    r_anl = map(func, t_anl)
    
    return t_anl, r_anl
    
def CosmologicalSolution():
    pass    