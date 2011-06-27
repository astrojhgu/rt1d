"""
Analysis.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2011-06-17.

Description: Functions to calculate various quantities from our rt1d datasets.
     
"""

import numpy as np
import pylab as pl

class Analysis:
    def __init__(self, pf):
        self.pf = pf
        
    