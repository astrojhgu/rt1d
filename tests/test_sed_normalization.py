"""

test_sed_normalization.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Apr  5 09:22:05 2013

Description: 

"""

import rt1d
import numpy as np

pf = \
{
 'source_type': 'diffuse',
 'spectrum_type': 'pl',
 'spectrum_fraction': 1.,
 'spectrum_alpha': -1.5,
 'spectrum_Emin': 2e2,
 'spectrum_Emax': 3e4,
}

rs = rt1d.sources.RadiationSource(init_tabs=False, **pf)

E = np.linspace(rs.Emin, rs.Emax, 10000)
LE = np.array(map(rs.Spectrum, E))

print 'Normalization accurate to %g for PL.' % (abs(1.-np.trapz(LE, x=E)))



