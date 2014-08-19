"""

Warnings.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Aug 13 14:31:51 MDT 2014

Description: 

"""

import sys
import numpy as np
from .PrintInfo import twidth, line, tabulate

separator = '|'*twidth
separator2 = '-'*twidth

def dt_error(grid, z, q, dqdt, new_dt, cell, method):
    
    print ""    
    print line(separator)
    print line('WARNING: something wrong with the time-step')    
    print line(separator)
    
    print line(separator2)    
    if new_dt <= 0:
        print line("current dt  : %.4e" % new_dt)
    else:
        print line("current dt  : NaN or inf")
                
    print line(separator2)
    print line("method      : %s" % method)
    print line("cell #      : %i" % cell)
    if z is not None:
        print line("redshift    : %.4g" % z)
    print line(separator2)  
     
    cols = ['value', 'derivative']
    
    rows = []
    data = []
    for i in range(len(grid.qmap)):
        name = grid.qmap[i]
        rows.append(name)
                
        data.append([q[i], dqdt[i]])
            
    # Print quantities and their rates of change
    tabulate(data, rows, cols, cwidth=12)  

    print line(separator2)        

    print line(separator)
    print ""
    
    sys.exit(1)
    
    
        