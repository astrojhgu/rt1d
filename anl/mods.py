"""
mods.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-12-01.

Description: Simple script to load up rtanl environment.  Should be aliased to 'irt', and should have referenced another
alias, 'rtpath', before beginning.
     
"""

import numpy as np
import pylab as pl
import os
from Constants import *
from rtanl import *

print "\nWelcome to irt!\n"

print "Standard rt1d analysis module has been loaded.  To get started: \n"
print "    mpf = rtanl(MainParameterFile)"
print "    sim = mpf.LoadDataset(parvals = {}, filename = None)\n"
