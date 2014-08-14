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

def dt_error(grid, q, dqdt, new_dt, which_cell, methods):
    
    print ""    
    print line(separator)
    print line('WARNING: something wrong with the time-step')    
    print line(separator)
    
    print line(separator2)    
    if new_dt <= 0:
        print line("current dt  : %.4e" % new_dt)
    else:
        print line("current dt  : NaN or inf")
            
    same_cell = len(np.unique(which_cell)) == 1
    
    for h, cell in enumerate(which_cell): 
        
        if same_cell:
            mth = ''
            for m in methods:
                mth += '%s, ' % m
            mth = mth.rstrip(',')
        else:
            mth = methods[h]
        
        print line(separator2)
        print line("method      : %s" % mth)
        print line("cell #      : %i" % cell)
        print line(separator2)   
        cols = ['value', 'derivative']
        
        rows = []
        data = []
        for i in range(len(grid.qmap)):
            name = grid.qmap[i]
            rows.append(name)
            
            data.append([q[0,i], dqdt[0,i]])
    
        
        # Print quantities and their rates of change
        tabulate(data, rows, cols, cwidth=12)  

        if same_cell:
            break
    
    print line(separator2)        

    print line(separator)
    print ""
    
    sys.exit(1)
    
    
        