"""

Hydrogen.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Mar 12 18:02:07 2012

Description: Container for hydrogen physics stuff.

"""

import numpy as np
from .Constants import *
from .Cosmology import *

class Hydrogen:
    def __init__(self, pf):
        self.pf = pf
        self.cosm = Cosmology(pf)
        #self.nmax = self.pf["LymanAlphaNmax"]
        
    def hydrogen_freq(self, nu, nl):
        return Ryd * (1. / nl / nl - 1. / nu / nu) / h

    def zmax(self, z, n):
        return (1. + z) * (1. - (n + 1)**-2) / (1. - n**-2) - 1.
    
    def frec(self, n):
        if n == 2: return 1.0
        if n == 3: return 0.0
        if n == 4: return 0.2609
        if n == 5: return 0.3078
        if n == 6: return 0.3259
        if n == 7: return 0.3353
        if n == 8: return 0.3410
        if n == 9: return 0.3448
        if n == 10: return 0.3476
        if n == 11: return 0.3496
        if n == 12: return 0.3512
        if n == 13: return 0.3524
        if n == 14: return 0.3535
        if n == 15: return 0.3543
        if n == 16: return 0.3550
        if n == 17: return 0.3556
        if n == 18: return 0.3561
        if n == 19: return 0.3565
        if n == 20: return 0.3569
        if n == 21: return 0.3572
        if n == 22: return 0.3575
        if n == 23: return 0.3578
        if n == 24: return 0.3580
        if n == 25: return 0.3582
        if n == 26: return 0.3584
        if n == 27: return 0.3586
        if n == 28: return 0.3587
        if n == 29: return 0.3589    
        if n == 30: return 0.3590
    
    # Look at line 905 in astrophysics.cc of jonathan's code
    
    def BrightnessTemperature(self, z, xHII, delta, Ts):
        """
        Global 21-cm signature relative to cosmic microwave background in mK.
        """
        
        return 27. * (1. - xHII) * (1.0 + delta) * \
            (self.cosm.OmegaBaryonNow * self.cosm.h_70**2 / 0.023) * \
            (0.15 * (1.0 + z) / self.cosm.OmegaMatterNow / self.cosm.h_70**2 / 10.)**0.5 * \
            (1.0 - self.cosm.TCMB(z) / Ts)
            
            
            
            